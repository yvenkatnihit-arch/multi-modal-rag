from src.query.hybrid_search import SearchHit
from src.query.merger import merge, similarity


def hit(topic, n, text=None, score=0.03, dense_rank=None, dense_distance=None, bm25_rank=None, bm25_score=None):
    return SearchHit(
        id=f"{topic}/f.md/s{n}/0", topic_id=topic, text=text or f"unique text of {topic} chunk {n} about subject{topic}{n}",
        metadata={"topic_id": topic, "source": "f.md", "modality": "text"}, score=score,
        dense_rank=dense_rank, dense_distance=dense_distance, bm25_rank=bm25_rank, bm25_score=bm25_score,
    )


def ids(result):
    return [h.id.split("/")[0] + h.id.split("/s")[1].split("/")[0] for h in result.hits]   # 'a1', 'b2'...


def test_single_topic_keeps_order_and_adds_rank_fields_without_losing_scores():
    src = [hit("a", i, score=0.03 - i * 0.001, dense_rank=i, dense_distance=0.2 + i / 10, bm25_rank=i, bm25_score=9.0 - i) for i in range(1, 4)]
    r = merge({"a": src}, k=5)
    assert ids(r) == ["a1", "a2", "a3"]
    assert [h.topic_rank for h in r.hits] == [1, 2, 3]
    assert r.hits[0].merged_score == 1 / 61 and r.hits[1].merged_score == 1 / 62
    first = r.hits[0]                                                   # known issue #5: nothing is discarded
    keep = lambda h: (h.score, h.dense_rank, h.dense_distance, h.bm25_rank, h.bm25_score)
    assert keep(first) == keep(src[0])                                  # every original evidence field is untouched


def test_topics_are_interleaved_by_in_topic_rank():
    r = merge({"a": [hit("a", i) for i in (1, 2, 3)], "b": [hit("b", i) for i in (1, 2, 3)]}, k=6)
    assert [x[0] for x in ids(r)] == ["a", "b", "a", "b", "a", "b"]
    assert [h.topic_rank for h in r.hits] == [1, 1, 2, 2, 3, 3]


def test_equal_ranks_are_ordered_by_stronger_evidence_not_by_topic_order():
    weak, strong = hit("a", 1, score=0.016), hit("b", 1, score=0.033)
    assert ids(merge({"a": [weak], "b": [strong]}, k=2)) == ["b1", "a1"]
    tie_a, tie_b = hit("a", 1, score=0.02, dense_distance=0.40), hit("b", 1, score=0.02, dense_distance=0.25)
    assert ids(merge({"a": [tie_a], "b": [tie_b]}, k=2)) == ["b1", "a1"]       # then the closer dense match


def test_k_cuts_the_list_and_candidates_are_reported():
    r = merge({"a": [hit("a", i) for i in range(1, 6)], "b": [hit("b", i) for i in range(1, 6)]}, k=4)
    assert len(r.hits) == 4 and r.candidates == {"a": 5, "b": 5}


def test_identical_content_in_two_topics_keeps_only_the_better_ranked_copy():
    same = "the survey recorded fifty five snake sightings in mombasa during twenty twenty five and twelve relocations"
    r = merge({"a": [hit("a", 1, text=same)], "b": [hit("b", 1, text=same), hit("b", 2)]}, k=5)
    assert ids(r) == ["a1", "b2"] and r.duplicates == [("b/f.md/s1/0", "a/f.md/s1/0")]


def test_near_duplicate_is_dropped_but_table_rows_sharing_a_header_are_not():
    base = " ".join(f"word{i}" for i in range(60))
    near = base + " extra"                                              # one added word
    assert similarity(base, near) > 0.9
    header = "row | Restaurant Name | City | Aggregate rating | Votes"
    g1 = header + "\n" + "\n".join(f"{i} | Place{i} | City{i} | 4.{i} | {i * 7}" for i in range(1, 9))
    g2 = header + "\n" + "\n".join(f"{i} | Place{i} | City{i} | 3.{i} | {i * 5}" for i in range(9, 17))
    assert similarity(g1, g2) < 0.5
    r = merge({"a": [hit("a", 1, text=base), hit("a", 2, text=near), hit("a", 3, text=g1), hit("a", 4, text=g2)]}, k=10)
    assert ids(r) == ["a1", "a3", "a4"] and [d[0] for d in r.duplicates] == ["a/f.md/s2/0"]


def test_distance_floor_drops_weak_dense_only_hits_but_never_keyword_supported_ones():
    weak_dense_only = hit("a", 1, dense_rank=1, dense_distance=0.60)
    weak_but_keyword = hit("a", 2, dense_rank=2, dense_distance=0.60, bm25_rank=1, bm25_score=7.0)
    keyword_only = hit("a", 3, bm25_rank=2, bm25_score=5.0)                       # not in the dense list at all
    close = hit("a", 4, dense_rank=3, dense_distance=0.20)
    r = merge({"a": [weak_dense_only, weak_but_keyword, keyword_only, close]}, k=10, max_distance=0.45)
    assert ids(r) == ["a2", "a3", "a4"] and r.too_far == [weak_dense_only.id]
    assert [h.topic_rank for h in r.hits] == [1, 2, 3]                            # ranks are assigned after the floor
    assert ids(merge({"a": [weak_dense_only]}, k=10)) == ["a1"]                   # off by default


def test_empty_inputs():
    assert merge({}).hits == []
    r = merge({"a": [], "b": [hit("b", 1)]}, k=3)
    assert ids(r) == ["b1"] and r.candidates == {"a": 0, "b": 1}


def test_result_is_deterministic():
    data = {"a": [hit("a", i, score=0.02) for i in range(1, 5)], "b": [hit("b", i, score=0.02) for i in range(1, 5)]}
    first = ids(merge(data, k=6))
    assert all(ids(merge(data, k=6)) == first for _ in range(5))
    assert ids(merge(dict(reversed(list(data.items()))), k=6)) != first      # the router's topic order is the final tie-break
