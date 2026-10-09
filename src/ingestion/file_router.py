"""File router: decides which parser handles each file in a topic folder.

Detection uses the extension AND the file's content (magic bytes / a peek inside), because
names lie. Unsupported or unreadable files are skipped and logged with a reason.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.core.topic_registry import Topic

log = logging.getLogger(__name__)

Kind = Literal["pdf", "image", "table", "text"]

EXT_KIND: dict[str, Kind] = {
    ".pdf": "pdf",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".csv": "table", ".xlsx": "table", ".json": "table",   # json may be demoted to text
    ".txt": "text", ".md": "text", ".html": "text", ".htm": "text", ".docx": "text", ".log": "text",
}
IGNORED_NAMES = {"topic.json", "readme.md", "thumbs.db", ".ds_store"}
MAX_SNIFF = 8192


@dataclass(frozen=True)
class RoutedFile:
    path: Path
    source: str          # path relative to the topic folder, forward slashes
    kind: Kind
    note: str = ""       # set when content overrode the extension


@dataclass(frozen=True)
class SkippedFile:
    source: str
    reason: str


def _magic_kind(head: bytes) -> str | None:
    """Identify binary formats from their first bytes. Returns None for plain text."""
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff") or (
        head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    ):
        return "image"
    if head.startswith(b"PK\x03\x04"):
        return "zip"        # xlsx and docx are both zip containers
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "ole"        # legacy .xls/.doc
    return None


def _json_kind(path: Path) -> Kind | None:
    """A JSON file is a table if it holds a list of objects (at top level or one level down)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    def is_records(v):
        return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v[:20])
    if is_records(data):
        return "table"
    if isinstance(data, dict) and any(is_records(v) for v in data.values()):
        return "table"
    return "text"


def route_file(path: Path, topic_dir: Path) -> RoutedFile | SkippedFile:
    source = path.relative_to(topic_dir).as_posix()
    ext = path.suffix.lower()
    try:
        size = path.stat().st_size
        if size == 0:
            return SkippedFile(source, "empty file")
        with open(path, "rb") as f:
            head = f.read(MAX_SNIFF)
    except OSError as e:
        return SkippedFile(source, f"unreadable: {e}")

    if ext not in EXT_KIND:
        return SkippedFile(source, f"unsupported extension {ext or '(none)'}")
    kind = EXT_KIND[ext]
    magic = _magic_kind(head)
    note = ""

    # Content wins over the extension.
    if magic in ("pdf", "image"):
        if magic != kind:
            note = f"extension {ext} but content is {magic}"
            kind = magic
    elif magic == "ole":
        return SkippedFile(source, "legacy OLE format (.xls/.doc) is not supported")
    elif magic == "zip":
        if ext not in (".xlsx", ".docx"):
            return SkippedFile(source, f"zip container with extension {ext}")
    else:  # no binary signature -> must be plain text
        if kind in ("pdf", "image") or ext in (".xlsx", ".docx"):
            return SkippedFile(source, f"{ext} file is not actually a {ext[1:]} (corrupt or renamed)")
        if b"\x00" in head:
            return SkippedFile(source, "binary content in a text-type file")
        if ext == ".json":
            kind = _json_kind(path)
            if kind is None:
                return SkippedFile(source, "invalid JSON")
            if kind == "text":
                note = "JSON without a list of records is treated as text"
    return RoutedFile(path, source, kind, note)


def route_topic(topic: Topic) -> tuple[list[RoutedFile], list[SkippedFile]]:
    routed: list[RoutedFile] = []
    skipped: list[SkippedFile] = []
    for p in sorted(topic.path.rglob("*")):
        if not p.is_file():
            continue
        if p.name.lower() in IGNORED_NAMES or p.name.startswith(("~$", ".")):
            continue
        r = route_file(p, topic.path)
        if isinstance(r, SkippedFile):
            skipped.append(r)
            log.warning("[%s] skipped %s: %s", topic.id, r.source, r.reason)
        else:
            routed.append(r)
            if r.note:
                log.info("[%s] %s -> %s (%s)", topic.id, r.source, r.kind, r.note)
    return routed, skipped


if __name__ == "__main__":
    from collections import Counter
    from src.core.topic_registry import list_topics

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for t in list_topics():
        routed, skipped = route_topic(t)
        counts = Counter(r.kind for r in routed)
        print(f"{t.id:18} {len(routed):3} routed {dict(counts)}  skipped {len(skipped)}")
