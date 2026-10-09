"""The normalized record: the one shape every parser returns.

Everything downstream (chunker, embeddings, search, context, citations) only knows
about Record, never about PDFs, CSVs or JPGs.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

Modality = Literal["text", "table", "image"]
MODALITIES = ("text", "table", "image")


@dataclass(frozen=True)
class Metadata:
    topic_id: str
    source: str                       # file path relative to the topic folder, e.g. "images/15970.jpg"
    modality: Modality
    page: int | None = None           # 1-based page (PDF); None for files without pages
    table_id: str | None = None       # which stored table: "main" (csv/json), a sheet name, "p3t1" (pdf)
    part: str | None = None           # which piece of that table: "summary", "rows-1-23"
    image_path: str | None = None     # saved copy under assets/, so the generator can show the image
    heading: str | None = None        # section heading the text sits under, for citations

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {self.modality!r}")
        if not self.topic_id or not self.source:
            raise ValueError("topic_id and source are required")

    def to_dict(self) -> dict[str, Any]:
        """Chroma metadata accepts only str/int/float/bool, so None values are dropped."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Metadata":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Record:
    text: str
    metadata: Metadata
    extra: dict[str, Any] = field(default_factory=dict, compare=False)  # parser-private notes, never stored

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError(f"empty text for {self.metadata.source}")

    @property
    def label(self) -> str:
        return make_label(self.metadata)


_PDF_TABLE_ID = re.compile(r"^p(\d+)t(\d+)$")


def pdf_table_ref(table_id: str | None) -> tuple[int, int] | None:
    """Tables extracted from a PDF are named 'p<page>t<n>': 'p3t2' -> (3, 2). Anything else -> None."""
    m = _PDF_TABLE_ID.match(table_id or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def make_label(m: Metadata) -> str:
    """Short human label, e.g. 'text p.3', 'table sales.csv', 'image fig2.png'."""
    name = m.source.rsplit("/", 1)[-1]
    if m.modality == "text":
        return f"text p.{m.page}" if m.page else f"text {name}"
    if m.modality == "table":
        label = f"table {name}"
        pdf = pdf_table_ref(m.table_id)
        if pdf:
            label += f" p.{pdf[0]} t{pdf[1]}"
        elif m.table_id and m.table_id != "main":
            label += f" [{m.table_id}]"
        if m.part:
            if m.part.startswith("rows-"):
                label += " " + m.part.replace("rows-", "rows ", 1)
            elif m.part.startswith("sample-"):
                label += " " + m.part.replace("sample-", "sample rows ", 1)
            else:
                label += f" ({m.part})"
        return label
    return f"image {name}" + (f" p.{m.page}" if m.page else "")


@dataclass(frozen=True)
class Chunk:
    """A Record-sized-for-embedding: what actually goes into the vector store."""
    id: str                  # stable: topic/source/location/n
    text: str
    metadata: Metadata

    @property
    def label(self) -> str:
        return make_label(self.metadata)

    @property
    def content_hash(self) -> str:
        """Fingerprint of text + metadata. Changes only when either changes, so re-ingestion
        can skip unchanged chunks (no re-embedding, no API cost)."""
        payload = json.dumps([self.text, self.metadata.to_dict()], sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
