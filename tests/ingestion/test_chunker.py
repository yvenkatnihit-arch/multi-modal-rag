import pytest

from src import config
from src.ingestion.chunker import chunk_records
from src.core.records import Metadata, Record


def rec(text, modality="text", source="a.md", topic="t", **meta):
    return Record(text, Metadata(topic, source, modality, **meta))


def sentences(n):
    return " ".join(f"Sentence number {i:03d} talks about snakes." for i in range(n))


def test_short_record_is_one_chunk_with_stable_id():
    [c] = chunk_records([rec("Hello world.", page=3, source="r.pdf")])
    assert c.id == "t/r.pdf/p3/0" and c.text == "Hello world."


def test_long_text_is_split_within_size_and_overlaps():
    chunks = chunk_records([rec(sentences(120))])
    assert len(chunks) > 3
    assert all(len(c.text) <= config.CHUNK_SIZE for c in chunks)
    for a, b in zip(chunks, chunks[1:]):          # a sentence repeats across each boundary
        tail = a.text.split(". ")[-2:]
        assert any(s.strip(" .") in b.text for s in tail if s.strip())
    assert [c.id for c in chunks] == [f"t/a.md/s0/{i}" for i in range(len(chunks))]


def test_heading_is_repeated_on_every_chunk_and_size_still_respected():
    chunks = chunk_records([rec(sentences(60), heading="Report > Risk")])
    assert len(chunks) > 1
    assert all(c.text.startswith("Report > Risk\n\n") for c in chunks)
    assert all(len(c.text) <= config.CHUNK_SIZE for c in chunks)


def test_separator_only_tail_piece_is_dropped():
    body = sentences(30)                       # long enough to split; ends with a markdown rule
    chunks = chunk_records([rec(body + "\n\n---", heading="Report > Risk")])
    assert all(any(ch.isalnum() for ch in c.text.split("\n\n", 1)[1]) for c in chunks)


def test_ids_are_stable_across_runs_and_unaffected_by_other_files():
    a = [rec("Alpha text.", source="a.md"), rec("More alpha.", source="a.md")]
    b = [rec("Beta text.", source="b.md")]
    ids1 = [c.id for c in chunk_records(a)]
    ids2 = [c.id for c in chunk_records(b + a)]          # a different file added first
    assert ids1 == ["t/a.md/s0/0", "t/a.md/s1/0"]
    assert all(i in ids2 for i in ids1)


def test_hash_changes_only_with_text():
    [c1] = chunk_records([rec("same")])
    [c2] = chunk_records([rec("same", source="a.md")])
    [c3] = chunk_records([rec("different")])
    assert c1.content_hash == c2.content_hash != c3.content_hash


def test_table_split_repeats_header_and_keeps_every_row():
    header = "city | rating | votes"
    rows = [f"city{i:04d} | {i % 5} | {i * 3}" for i in range(400)]
    chunks = chunk_records([rec("\n".join([header, *rows]), "table", source="z.csv", table_id="rows-0-399")])
    assert len(chunks) > 1
    assert all(c.text.startswith(header) for c in chunks)
    assert all(len(c.text) <= config.TABLE_MAX_CHARS for c in chunks)
    kept = [l for c in chunks for l in c.text.split("\n")[1:]]
    assert kept == rows
    assert chunks[0].id == "t/z.csv/trows-0-399/0"


def test_small_table_and_image_stay_whole():
    [t] = chunk_records([rec("a | b\n1 | 2", "table", source="x.csv", table_id="summary")])
    [i] = chunk_records([rec("Receipt total 193.00 " * 20, "image", source="r.jpg", image_path="assets/t/r.jpg")])
    assert t.text == "a | b\n1 | 2"
    assert i.id == "t/r.jpg/ir/0" and len(i.text) < config.IMAGE_MAX_CHARS


def test_two_images_on_one_pdf_page_get_distinct_ids():
    chunks = chunk_records([
        rec("fig one", "image", source="p.pdf", page=2, image_path="assets/t/p-2-1.png"),
        rec("fig two", "image", source="p.pdf", page=2, image_path="assets/t/p-2-2.png"),
    ])
    assert len({c.id for c in chunks}) == 2


def test_colliding_locations_raise():
    with pytest.raises(ValueError, match="duplicate chunk id"):
        chunk_records([rec("x", page=1, source="p.pdf"), rec("y", page=1, source="p.pdf")])
