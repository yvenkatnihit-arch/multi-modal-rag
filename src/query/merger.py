"""Merger: one ranked list per topic  ->  one list for the generator.

    topic A: a1 a2 a3          cross-topic RRF (each hit scores 1/(60 + rank in its own topic))
    topic B: b1 b2 b3    -->   a1 b1 a2 b2 a3 b3, near-duplicates removed, cut to k

Interleaving by in-topic rank keeps every routed topic represented, which a multi-topic question needs.
Nothing is thrown away: each hit keeps its dense distance, BM25 score, both ranks and its topic-local
score, and gains `topic_rank` and `merged_score`. Ties between topics go to the stronger evidence.
"""
import re
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from src import config
from src.query.hybrid_search import RRF_K, SearchHit

DEDUP_SIMILARITY = 0.9       # shingle overlap at or above this = the same content
SHINGLE = 3                  # words per shingle: order-sensitive, so table row groups with different rows differ


@dataclass(frozen=True)
class MergeResult:
    hits: list[SearchHit]
    duplicates: list[tuple[str, str]] = field(default_factory=list)   # (dropped chunk id, kept chunk id)
    too_far: list[str] = field(default_factory=list)                  # ids removed by the max_distance floor
    candidates: dict[str, int] = field(default_factory=dict)          # topic -> hits it contributed before merging


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    if len(words) <= SHINGLE:
        return {tuple(words)}
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of word-3-gram sets: 1.0 = same content, 0 = nothing shared."""
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def merge(
    results: Mapping[str, Sequence[SearchHit]],
    k: int | None = None,
    max_distance: float | None = None,
    dedup_similarity: float = DEDUP_SIMILARITY,
) -> MergeResult:
    """`results` maps topic id -> that topic's hits, best first (the order the router listed the topics is kept)."""
    k = k or config.TOP_K
    topic_order = {t: i for i, t in enumerate(results)}
    too_far: list[str] = []

    scored: list[SearchHit] = []
    for topic, hits in results.items():
        kept = []
        for h in hits:
            # a weak meaning match with no keyword support is noise; a keyword match is never dropped
            if max_distance is not None and h.dense_distance is not None and h.dense_distance > max_distance and h.bm25_rank is None:
                too_far.append(h.id)
            else:
                kept.append(h)
        for rank, h in enumerate(kept, 1):
            scored.append(replace(h, topic_rank=rank, merged_score=1.0 / (RRF_K + rank)))

    scored.sort(key=lambda h: (
        -h.merged_score,                                   # interleave topics by in-topic rank
        -h.score,                                          # then stronger evidence first
        h.dense_distance if h.dense_distance is not None else float("inf"),
        topic_order.get(h.topic_id, 0),
        h.id,
    ))

    final: list[SearchHit] = []
    shingles: dict[str, set] = {}
    duplicates: list[tuple[str, str]] = []
    for h in scored:
        sh = _shingles(h.text)
        twin = next((kept for kept in final if _jaccard(sh, shingles[kept.id]) >= dedup_similarity), None)
        if twin is not None:
            duplicates.append((h.id, twin.id))
            continue
        final.append(h)
        shingles[h.id] = sh
        if len(final) == k:
            break

    return MergeResult(final, duplicates, too_far, {t: len(h) for t, h in results.items()})


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0
