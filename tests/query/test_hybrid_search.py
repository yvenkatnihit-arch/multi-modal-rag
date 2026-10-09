import numpy as np
import pytest

from src.query import hybrid_search as hs
from src.query.hybrid_search import HybridSearcher, rrf_scores, tokenize
from src.core.vector_store import VectorStore
from tests.helpers import Embedder, chunk, fake_vec


def unit(i, dim=8):
    v = [0.0] * dim
    v[i] = 1.0
    return v


class OneHotEmbedder:
    """Lets a test decide exactly where each chunk sits in vector space."""

    def __init__(self, by_text):
        self.by_text = by_text

    def __call__(self, texts):
        return [self.by_text[t] for t in texts]


FILLER = [chunk(i, f"filler{i} word{i} other{i}") for i in range(10, 16)]


def searcher(tmp_path, chunks, embed=None):
    store = VectorStore(tmp_path / "chroma", embed_fn=embed or Embedder())
    store.sync_topic("t", chunks)
    return HybridSearcher(store), store


# ---------------------------------------------------------------- the pure pieces
def test_tokenize_folds_case_and_accents_and_keeps_ids_whole():
    assert tokenize("São Paulo, X51005361900! ünï") == ["sao", "paulo", "x51005361900", "uni"]


def test_keyword_tokens_split_camel_case_strip_plurals_and_leave_identifiers_alone():
    assert hs.bm25_tokens("baseColour productDisplayName") == ["base", "colour", "product", "display", "name"]
    assert [hs.stem(w) for w in ["skirts", "cities", "boxes", "watches", "class", "status", "analysis", "bus", "2025s", "x123"]] == \
           ["skirt", "city", "box", "watch", "class", "status", "analysis", "bus", "2025s", "x123"]
    assert hs.bm25_tokens("Pleated Black Skirts") == hs.bm25_tokens("pleated black skirt")
    assert hs.is_identifier("12839") and hs.is_identifier("x51005361900") and not hs.is_identifier("skirt") and not hs.is_identifier("42")


def test_a_repeated_question_word_counts_once_and_identifiers_count_extra():
    assert hs.query_tokens("colour of the colour of 12839") == ["colour"] + ["12839"] * hs.ID_BOOST


def test_a_rare_identifier_outranks_generic_words_in_the_keyword_leg(tmp_path):
    """The real failure: the chunk holding product 12839's row matched only the id, while chunks full of 'colour',
    'skirt' and 'catalog' outscored it."""
    generic = [chunk(i, f"base colour skirt catalog colour skirts list product photo item{i} " * 3) for i in range(10)]
    target = chunk(50, "row 114 | 12839 | Women | Apparel | Bottomwear | Black | Fall | 2011 | Ant Kids Pleated Black")
    image = chunk(51, "Image 12839.jpg (photo). A light grey pleated skirt.")
    s, _ = searcher(tmp_path, generic + [target, image])
    idx = s._index("t")
    ranking = s._bm25_ranking(idx, "What colour is the skirt in product photo 12839, and what colour does the catalog list for it?", 24, None)
    order = [idx.ids[i] for i, _ in ranking]                           # the keyword leg's own ranking, best first
    assert order.index("t/a.md/s50/0") < 2 and order.index("t/a.md/s51/0") < 2     # both chunks that hold the id lead


def test_plural_and_singular_forms_find_each_other(tmp_path):
    s, _ = searcher(tmp_path, [chunk(0, "Ant Kids Pleated Black Skirts")] + [chunk(i, f"unrelated words{i} here{i}") for i in range(1, 9)])
    hit = s.search("t", "skirt", k=3, embed_fn=fake_vec)[0]
    assert hit.id.endswith("s0/0") and hit.bm25_rank == 1


def test_keyword_leg_ignores_stopwords(tmp_path):
    assert hs.bm25_tokens("a song about the girl who smiles") == ["song", "girl", "smile"]
    s, _ = searcher(tmp_path, [chunk(i, f"the {w} is a thing about it") for i, w in enumerate("abcdefgh")])
    hits = s.search("t", "who is it about", k=3, embed_fn=fake_vec)      # nothing but stop-words
    assert hits and all(h.bm25_rank is None for h in hits)               # only the dense leg voted


def test_rrf_rewards_agreement_and_is_rank_based():
    s = rrf_scores([["a", "b", "c"], ["b", "d"]])
    assert s["b"] > s["a"] > s["c"]                                  # b is in both lists
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["d"] == pytest.approx(1 / 62)
    assert rrf_scores([["a"], ["a"]], weights=[2.0, 0.5])["a"] == pytest.approx(2.5 / 61)


# ---------------------------------------------------------------- fusion behaviour
def test_fusion_combines_a_dense_only_hit_with_a_keyword_hit(tmp_path):
    chunks = [chunk(0, "alpha beta gamma"), chunk(1, "delta epsilon"), chunk(2, "zeta unique7 eta"), *FILLER]
    vecs = {c.text: unit(3 + i % 5) for i, c in enumerate(FILLER)}
    vecs.update({"alpha beta gamma": unit(0), "delta epsilon": unit(1), "zeta unique7 eta": unit(2)})
    s, _ = searcher(tmp_path, chunks, OneHotEmbedder({**{c.text: vecs[c.text] for c in chunks}}))
    qvec = list(0.8 * np.array(unit(1)) + 0.6 * np.array(unit(2)))   # meaning points at chunk 1, then 2
    hits = s.search("t", "unique7", k=3, query_vector=qvec)

    by_id = {h.id: h for h in hits}
    one, two = by_id["t/a.md/s1/0"], by_id["t/a.md/s2/0"]
    assert one.dense_rank == 1 and one.bm25_rank is None             # only the meaning list likes it
    assert two.bm25_rank == 1 and two.dense_rank == 2                # both lists like it
    assert hits[0].id == two.id and two.score > one.score           # agreement wins
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_keyword_leg_finds_an_exact_id(tmp_path):
    chunks = [chunk(i, f"receipt number {i} for shop {i}") for i in range(8)] + [chunk(9, "invoice X51005361900 total 18.00")]
    s, _ = searcher(tmp_path, chunks)
    top = s.search("t", "X51005361900", k=3, embed_fn=fake_vec)[0]
    assert top.id == "t/a.md/s9/0" and top.bm25_rank == 1


def test_question_with_no_shared_words_still_gets_dense_results(tmp_path):
    s, _ = searcher(tmp_path, [chunk(i, f"topic{i} text{i} item{i}") for i in range(8)])
    hits = s.search("t", "zzzz qqqq", k=3, embed_fn=fake_vec)
    assert hits and all(h.bm25_rank is None for h in hits)


def test_modality_filter_applies_to_both_lists(tmp_path):
    chunks = [chunk(0, "snake sightings narrative text"), chunk(1, "snake sightings table", modality="table", source="s.csv", table_id="main", part="summary")]
    chunks += [chunk(i, f"other{i} words{i} here{i}") for i in range(2, 10)]
    s, _ = searcher(tmp_path, chunks)
    only_tables = s.search("t", "snake sightings", k=5, modality="table", embed_fn=fake_vec)
    assert [h.metadata["modality"] for h in only_tables] == ["table"] * len(only_tables) and only_tables
    assert {h.metadata["modality"] for h in s.search("t", "snake sightings", k=5, embed_fn=fake_vec)} >= {"text", "table"}


def test_hits_carry_label_and_trace_fields(tmp_path):
    s, _ = searcher(tmp_path, [chunk(0, "hello world", page=3, source="r.pdf")] + FILLER)
    [h] = s.search("t", "hello world", k=1, embed_fn=fake_vec)
    assert h.label == "text p.3" and h.topic_id == "t" and h.dense_distance is not None and h.bm25_score > 0


def test_empty_or_unknown_topic_returns_nothing(tmp_path):
    s, _ = searcher(tmp_path, [chunk(0, "x")])
    assert s.search("nothing_here", "x", embed_fn=fake_vec) == []


def test_k_limits_results_and_results_are_deterministic(tmp_path):
    s, _ = searcher(tmp_path, [chunk(i, f"common word number{i}") for i in range(12)])
    a = s.search("t", "common word", k=4, embed_fn=fake_vec)
    b = s.search("t", "common word", k=4, embed_fn=fake_vec)
    assert len(a) == 4 and [h.id for h in a] == [h.id for h in b]


# ---------------------------------------------------------------- the cache (known issue #2)
def test_bm25_index_is_built_once_and_rebuilt_only_when_the_topic_changes(tmp_path):
    chunks = [chunk(i, f"text number{i} word{i}") for i in range(8)]
    s, store = searcher(tmp_path, chunks)
    for _ in range(5):
        s.search("t", "text word", embed_fn=fake_vec)
    assert s.builds == 1                                              # five queries, one build

    store.sync_topic("t", chunks)                                     # no-op sync
    s.search("t", "text word", embed_fn=fake_vec)
    assert s.builds == 1

    store.sync_topic("t", chunks + [chunk(20, "brand new chunk")])    # a chunk added
    assert any(h.id.endswith("s20/0") for h in s.search("t", "brand new chunk", k=1, embed_fn=fake_vec))
    assert s.builds == 2

    edited = chunks[:7] + [chunk(7, "text number7 EDITED")]            # same count, one chunk's text changed
    store.sync_topic("t", edited + [chunk(20, "brand new chunk")])
    s.search("t", "edited", embed_fn=fake_vec)
    assert s.builds == 3


def test_another_process_updating_the_data_is_noticed(tmp_path):
    s, _ = searcher(tmp_path, [chunk(i, f"text{i} word{i}") for i in range(8)])
    s.search("t", "text0", embed_fn=fake_vec)
    other = VectorStore(tmp_path / "chroma", embed_fn=Embedder())     # e.g. an ingestion run in another process
    other.sync_topic("t", [chunk(i, f"text{i} word{i}") for i in range(8)] + [chunk(50, "late arrival zebra")])
    assert s.search("t", "zebra", k=1, embed_fn=fake_vec)[0].id.endswith("s50/0")
