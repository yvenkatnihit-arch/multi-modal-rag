"""Hybrid search inside one topic: dense (meaning) + BM25 (exact words), fused with RRF.

    dense  : embed the question, find the nearest chunk vectors      -> good at paraphrase
    BM25   : score chunks by shared rare words (IDs, names, numbers) -> good at exact tokens
    RRF    : score = sum over lists of weight / (60 + rank). Uses ranks, not scores, because the
             two score scales are not comparable. A chunk liked by both lists beats one liked by one.

The BM25 index is built once per topic and cached; it is rebuilt only when the topic's stored
chunks change (detected by VectorStore.fingerprint), not on every query.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.core.embeddings import embed_query
from src.core.records import Metadata, make_label
from src.core.vector_store import VectorStore

RRF_K = 60
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase, accents folded, split into word/number tokens ('X51005361900' stays whole)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return _TOKEN.findall("".join(c for c in text if not unicodedata.combining(c)))


STOPWORDS = frozenset("""a an the and or but if of to in on at by for with about as from into than that this these those
it its is are was were be been being am do does did have has had having i you he she we they me my your our their who whom
what which when where why how can could should would will shall may might must not no so such there here any all some more
most other very just also then""".split())


_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
ID_BOOST = 3                       # an identifier in the question counts this many times as an ordinary word


def stem(token: str) -> str:
    """A deliberately small plural-stripper, applied identically to chunks and questions, so 'skirts' matches 'skirt'
    and 'cities' matches 'city'. Tokens with digits (ids, years) and short words are left alone."""
    if len(token) <= 3 or any(c.isdigit() for c in token):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if re.search(r"(ch|sh|x|z|ss)es$", token):
        return token[:-2]
    if token.endswith(("ss", "us", "is")):
        return token
    return token[:-1] if token.endswith("s") else token


def is_identifier(token: str) -> bool:
    """Codes, ids and numbers ('12839', 'x51005361900'): the most selective words in a question."""
    return len(token) >= 3 and any(c.isdigit() for c in token)


def bm25_tokens(text: str) -> list[str]:
    """Tokens for the keyword leg: camelCase split ('baseColour' -> 'base colour'), stop-words dropped, plurals stripped.
    A question like 'a song about a girl who smiles' is matched on 'song', 'girl', 'smile' and not on words that carry
    no meaning; meaning-only matching is the dense leg's job."""
    return [stem(t) for t in tokenize(_CAMEL.sub(" ", text)) if t not in STOPWORDS]


def query_tokens(question: str) -> list[str]:
    """What BM25 is asked: each distinct word once (a repeated word must not count twice), identifiers boosted."""
    distinct = list(dict.fromkeys(bm25_tokens(question)))
    return [t for t in distinct for _ in range(ID_BOOST if is_identifier(t) else 1)]


def rrf_scores(rankings: Sequence[Sequence[str]], weights: Sequence[float] | None = None, k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion. Each ranking is a best-first list of ids."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (k + rank)
    return scores


@dataclass(frozen=True)
class SearchHit:
    id: str
    topic_id: str
    text: str
    metadata: dict
    score: float                       # fused RRF score, higher is better
    dense_rank: int | None = None      # 1 = best; None = the dense list did not contain it
    dense_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    topic_rank: int | None = None      # set by the merger: position within its own topic's list (1 = best)
    merged_score: float | None = None  # set by the merger: cross-topic RRF score; `score` stays the topic-local one

    @property
    def label(self) -> str:
        return make_label(Metadata.from_dict(self.metadata))


class _TopicIndex:
    """All chunks of one topic plus their BM25 index."""

    def __init__(self, fingerprint: str, hits):
        self.fingerprint = fingerprint
        self.ids = [h.id for h in hits]
        self.texts = [h.text for h in hits]
        self.metadatas = [h.metadata for h in hits]
        self.modalities = np.array([m.get("modality") for m in self.metadatas])
        corpus = [bm25_tokens(t) for t in self.texts]
        self.bm25 = BM25Okapi(corpus) if any(corpus) else None      # BM25Okapi cannot handle an all-empty corpus


class HybridSearcher:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self._indexes: dict[str, _TopicIndex] = {}
        self.builds = 0                                             # how many BM25 indexes were built (for tests / tracing)

    def _index(self, topic_id: str) -> _TopicIndex:
        fp = self.store.fingerprint(topic_id)
        idx = self._indexes.get(topic_id)
        if idx is None or idx.fingerprint != fp:
            idx = _TopicIndex(fp, self.store.get_all(topic_id))
            self._indexes[topic_id] = idx
            self.builds += 1
        return idx

    def _bm25_ranking(self, idx: _TopicIndex, query: str, n: int, modality: str | None):
        tokens = query_tokens(query)
        if idx.bm25 is None or not tokens:
            return []
        scores = idx.bm25.get_scores(tokens)
        allowed = (idx.modalities == modality) if modality else np.ones(len(scores), dtype=bool)
        order = [i for i in np.argsort(-scores, kind="stable") if allowed[i] and scores[i] > 0]   # no shared word = no vote
        return [(int(i), float(scores[i])) for i in order[:n]]

    def search(
        self,
        topic_id: str,
        query: str,
        k: int | None = None,
        modality: str | None = None,
        query_vector: list[float] | None = None,
        weights: tuple[float, float] = (1.0, 1.0),
        embed_fn: Callable[[str], list[float]] = embed_query,
    ) -> list[SearchHit]:
        """Top-k chunks of one topic for a (standalone) question. Pass query_vector to avoid re-embedding
        the same question once per topic."""
        k = k or config.TOP_K
        n = max(k * 4, 20)                                          # candidates taken from each list before fusing
        idx = self._index(topic_id)
        if not idx.ids:
            return []
        qvec = query_vector if query_vector is not None else embed_fn(query)

        dense = self.store.dense_search(topic_id, qvec, n, modality)
        bm25 = self._bm25_ranking(idx, query, n, modality)

        dense_info = {h.id: (rank, h) for rank, h in enumerate(dense, 1)}
        position = {cid: i for i, cid in enumerate(idx.ids)}
        bm25_info = {idx.ids[i]: (rank, s) for rank, (i, s) in enumerate(bm25, 1)}

        fused = rrf_scores([[h.id for h in dense], [idx.ids[i] for i, _ in bm25]], list(weights))
        # best fused score first; ties go to the better dense rank, then id, so results are deterministic
        ordered = sorted(fused, key=lambda cid: (-fused[cid], dense_info.get(cid, (10**9,))[0], cid))[:k]

        hits = []
        for cid in ordered:
            d_rank, d_hit = dense_info.get(cid, (None, None))
            b_rank, b_score = bm25_info.get(cid, (None, None))
            i = position[cid]
            hits.append(SearchHit(
                id=cid, topic_id=topic_id, text=idx.texts[i], metadata=idx.metadatas[i], score=fused[cid],
                dense_rank=d_rank, dense_distance=d_hit.distance if d_hit else None,
                bm25_rank=b_rank, bm25_score=b_score,
            ))
        return hits
