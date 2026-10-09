import pytest

from src.core.vector_store import VectorStore, collection_name
from tests.helpers import Embedder, chunk, fake_vec


@pytest.fixture
def store(tmp_path):
    emb = Embedder()
    s = VectorStore(tmp_path / "chroma", embed_fn=emb)
    s.emb = emb
    return s


def test_collection_name_is_valid_for_short_ids():
    assert collection_name("ab") == "topic_ab"


def test_first_sync_adds_everything(store):
    stats = store.sync_topic("t", [chunk(0, "snakes in nairobi"), chunk(1, "pizza in rome")])
    assert (stats.added, stats.updated, stats.unchanged, stats.deleted) == (2, 0, 0, 0)
    assert store.count("t") == 2 and store.emb.texts_embedded == 2


def test_resync_unchanged_makes_zero_embedding_calls(store):
    chunks = [chunk(0, "snakes in nairobi"), chunk(1, "pizza in rome")]
    store.sync_topic("t", chunks)
    stats = store.sync_topic("t", chunks)
    assert stats.unchanged == 2 and stats.added == stats.updated == stats.deleted == 0
    assert store.emb.texts_embedded == 2          # still just the first run's


def test_changed_text_or_metadata_is_re_embedded_others_untouched(store):
    store.sync_topic("t", [chunk(0, "one two"), chunk(1, "three four"), chunk(2, "five six")])
    stats = store.sync_topic("t", [chunk(0, "one two"), chunk(1, "three four CHANGED"),
                                   chunk(2, "five six", heading="New heading")])
    assert stats.updated == 2 and stats.unchanged == 1
    assert store.emb.texts_embedded == 3 + 2
    docs = {h.id: h.text for h in store.get_all("t")}
    assert docs["t/a.md/s1/0"] == "three four CHANGED"


def test_removed_chunks_are_deleted_and_only_those(store):
    store.sync_topic("t", [chunk(0, "keep me"), chunk(1, "drop me")])
    stats = store.sync_topic("t", [chunk(0, "keep me")])
    assert stats.deleted == 1 and store.count("t") == 1
    assert store.get_all("t")[0].text == "keep me"


def test_adding_a_file_does_not_disturb_existing_chunks(store):
    store.sync_topic("t", [chunk(0, "old file text", source="a.md")])
    before = store.emb.texts_embedded
    stats = store.sync_topic("t", [chunk(0, "old file text", source="a.md"), chunk(0, "new file text", source="b.md")])
    assert (stats.added, stats.unchanged, stats.deleted) == (1, 1, 0)
    assert store.emb.texts_embedded == before + 1


def test_empty_chunk_list_cannot_wipe_a_populated_topic(store):
    store.sync_topic("t", [chunk(0, "precious")])
    with pytest.raises(ValueError, match="refusing to wipe"):
        store.sync_topic("t", [])
    assert store.count("t") == 1
    assert store.sync_topic("t", [], allow_empty=True).deleted == 1


def test_protected_sources_survive_a_failed_parse(store):
    store.sync_topic("t", [chunk(0, "from good file", source="good.md"), chunk(0, "from flaky file", source="flaky.pdf")])
    stats = store.sync_topic("t", [chunk(0, "from good file", source="good.md")], protect_sources={"flaky.pdf"})
    assert stats.deleted == 0 and stats.protected == 1 and store.count("t") == 2


def test_chunks_from_another_topic_are_rejected(store):
    with pytest.raises(ValueError, match="another topic"):
        store.sync_topic("t", [chunk(0, "x", topic="other")])


def test_dense_search_ranks_by_meaning_and_filters_by_modality(store):
    store.sync_topic("t", [
        chunk(0, "snakes relocated in nairobi kenya"),
        chunk(1, "pizza dough recipe from rome"),
        chunk(2, "snake sightings table nairobi", modality="table", source="s.csv", table_id="summary"),
    ])
    hits = store.dense_search("t", fake_vec("snakes in nairobi"), k=3)
    assert hits[0].id == "t/a.md/s0/0" and hits[0].distance < hits[-1].distance
    only_tables = store.dense_search("t", fake_vec("snakes in nairobi"), k=3, modality="table")
    assert [h.metadata["modality"] for h in only_tables] == ["table"]
    assert only_tables[0].metadata["table_id"] == "summary"


def test_search_on_empty_topic_returns_nothing(store):
    assert store.dense_search("nothing_here", fake_vec("x"), k=3) == []


def test_fingerprint_changes_exactly_when_the_topic_changes(store):
    store.sync_topic("t", [chunk(0, "one"), chunk(1, "two")])
    base = store.fingerprint("t")
    assert store.fingerprint("t") == base                                   # stable when nothing changed
    store.sync_topic("t", [chunk(0, "one"), chunk(1, "two")])
    assert store.fingerprint("t") == base                                   # a no-op sync changes nothing
    store.sync_topic("t", [chunk(0, "one"), chunk(1, "two CHANGED")])
    changed = store.fingerprint("t")
    assert changed != base                                                  # edited text
    store.sync_topic("t", [chunk(0, "one"), chunk(1, "two CHANGED"), chunk(2, "three")])
    assert store.fingerprint("t") not in (base, changed)                    # added chunk
    store.sync_topic("t", [chunk(0, "one")])
    assert store.fingerprint("t") != changed                                # removed chunks


def test_topics_are_isolated(store):
    store.sync_topic("t", [chunk(0, "alpha")])
    store.sync_topic("u", [chunk(0, "beta", topic="u")])
    assert store.topics() == ["t", "u"]
    assert [h.text for h in store.get_all("u")] == ["beta"]
