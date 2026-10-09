from src.ingestion.file_router import RoutedFile, SkippedFile, route_topic
from src.core.topic_registry import Topic


def run(tmp_path, files: dict[str, bytes]):
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    routed, skipped = route_topic(Topic("t", "T", "d", tmp_path))
    return {r.source: r for r in routed}, {s.source: s for s in skipped}


def test_normal_files_route_by_extension(tmp_path):
    routed, skipped = run(tmp_path, {"a.md": b"# hi", "b.csv": b"x,y\n1,2", "c.log": b"line"})
    assert {k: v.kind for k, v in routed.items()} == {"a.md": "text", "b.csv": "table", "c.log": "text"}
    assert not skipped


def test_content_overrides_extension(tmp_path):
    routed, _ = run(tmp_path, {"really_pdf.csv": b"%PDF-1.4 rest", "photo.txt": b"\xff\xd8\xff\xe0data"})
    assert routed["really_pdf.csv"].kind == "pdf" and routed["really_pdf.csv"].note
    assert routed["photo.txt"].kind == "image"


def test_bad_files_are_skipped_with_reason(tmp_path):
    _, skipped = run(tmp_path, {
        "empty.txt": b"",
        "fake.pdf": b"I am just text",
        "run.exe": b"MZ\x90\x00",
        "old.xls": b"\xd0\xcf\x11\xe0abc",
        "broken.json": b"{oops",
        "bin.txt": b"abc\x00def",
    })
    assert set(skipped) == {"empty.txt", "fake.pdf", "run.exe", "old.xls", "broken.json", "bin.txt"}
    assert all(s.reason for s in skipped.values())


def test_json_table_vs_text(tmp_path):
    routed, _ = run(tmp_path, {
        "rows.json": b'[{"a":1},{"a":2}]',
        "nested.json": b'{"results_shown": 2, "restaurants": [{"a":1}]}',
        "config.json": b'{"a": 1}',
    })
    assert routed["rows.json"].kind == "table"
    assert routed["nested.json"].kind == "table"
    assert routed["config.json"].kind == "text"


def test_housekeeping_files_and_subfolders(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("hello")
    routed, skipped = run(tmp_path, {"topic.json": b"{}", ".DS_Store": b"x", "~$lock.docx": b"x"})
    assert set(routed) == {"sub/deep.txt"}          # forward-slash relative source
    assert not skipped                              # ignored silently, not "skipped"
