

"""Chunker: Records -> Chunks sized for embedding, with stable IDs.
Chunk id = topic / source / location / n
  location: p<page>, t<table_id>, i<image name>, or s<record ordinal> when none of those apply
  n       : position of the piece within its record
The id depends only on where a chunk lives, never on a global counter, so re-ingesting a file
yields the same ids and adding a file never renumbers the others.
"""
import re
from pathlib import PurePosixPath
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.core.records import Chunk, Metadata, Record

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
_WORDY = re.compile(r"\w")


def _splitter(size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=min(config.CHUNK_OVERLAP, size // 4),
        separators=_SEPARATORS, keep_separator="end", strip_whitespace=True,
    )


def _split_prose(text: str, size: int | None = None) -> list[str]:
    size = size or config.CHUNK_SIZE
    # drop pieces with no letters/digits (a stray '---' left at the end of a split section)
    return [p for p in _splitter(size).split_text(text) if _WORDY.search(p)]


def _split_text_record(rec: Record) -> list[str]:
    """Split prose, repeating the heading path at the top of every piece."""
    heading = (rec.metadata.heading or "")[: config.HEADING_PREFIX_MAX]
    if not heading:
        return _split_prose(rec.text)
    prefix = heading + "\n\n"
    if len(prefix) + len(rec.text) <= config.CHUNK_SIZE:
        return [prefix + rec.text]
    room = max(config.CHUNK_SIZE - len(prefix), 200)
    return [prefix + body for body in _split_prose(rec.text, room)]


def _split_table_record(rec: Record) -> list[str]:
    """Split by lines; the first line (header) is repeated in every piece."""
    if len(rec.text) <= config.TABLE_MAX_CHARS:
        return [rec.text]
    header, *lines = rec.text.split("\n")
    pieces, current, size = [], [], len(header)
    for line in lines:
        if current and size + len(line) + 1 > config.TABLE_MAX_CHARS:
            pieces.append("\n".join([header, *current]))
            current, size = [], len(header)
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join([header, *current]))
    return pieces


def _split_image_record(rec: Record) -> list[str]:
    if len(rec.text) <= config.IMAGE_MAX_CHARS:
        return [rec.text]
    return _split_prose(rec.text)


_SPLIT = {"text": _split_text_record, "table": _split_table_record, "image": _split_image_record}


def _location(m: Metadata, ordinal: int) -> str:
    parts = []
    if m.page:
        parts.append(f"p{m.page}")
    if m.table_id:
        parts.append(f"t{m.table_id}")
    if m.part:
        parts.append(m.part)
    if m.image_path:
        parts.append(f"i{PurePosixPath(m.image_path.replace(chr(92), '/')).stem}")
    return "-".join(parts) or f"s{ordinal}"


def chunk_records(records: Iterable[Record]) -> list[Chunk]:
    by_file: dict[tuple[str, str], list[Record]] = {}
    for rec in records:
        by_file.setdefault((rec.metadata.topic_id, rec.metadata.source), []).append(rec)

    chunks: list[Chunk] = []
    for (topic, source), file_records in by_file.items():
        for ordinal, rec in enumerate(file_records):     # ordinal = position among the file's records
            loc = _location(rec.metadata, ordinal)
            for n, piece in enumerate(_SPLIT[rec.metadata.modality](rec)):
                chunks.append(Chunk(f"{topic}/{source}/{loc}/{n}", piece, rec.metadata))

    seen: set[str] = set()
    for c in chunks:
        if c.id in seen:
            raise ValueError(f"duplicate chunk id {c.id!r}: two records of one file share a location")
        seen.add(c.id)
    return chunks
