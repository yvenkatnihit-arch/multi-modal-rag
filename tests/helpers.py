"""Shared test doubles: a free, deterministic stand-in for the Gemini embedding model."""
import hashlib
import math

from src.core.records import Chunk, Metadata

DIM = 64


def fake_vec(text: str) -> list[float]:
    """Bag-of-words hashed into 64 dims: texts sharing words land close together."""
    v = [0.0] * DIM
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Embedder:
    """Counts how many texts were 'embedded', so tests can prove unchanged chunks cost nothing."""

    def __init__(self):
        self.texts_embedded = 0

    def __call__(self, texts):
        self.texts_embedded += len(texts)
        return [fake_vec(t) for t in texts]


def chunk(i, text, source="a.md", topic="t", modality="text", **meta):
    return Chunk(f"{topic}/{source}/s{i}/0", text, Metadata(topic, source, modality, **meta))


def html_pdf(tmp_path, bodies, name="doc.pdf", css=""):
    """A real PDF with one page per HTML body (<img src='file.png'> may refer to files in tmp_path).
    Each page is built on its own and merged, so every body is exactly one page."""
    import pymupdf

    out = pymupdf.open()
    for i, body in enumerate(bodies):
        story = pymupdf.Story(html=f"<html><head>{css}</head><body>{body}</body></html>",
                              archive=pymupdf.Archive(str(tmp_path)))
        part = tmp_path / f"_page{i}.pdf"                      # one file per page: Windows will not overwrite an open one
        writer, box = pymupdf.DocumentWriter(str(part)), pymupdf.paper_rect("a4")
        more = 1
        while more:
            dev = writer.begin_page(box)
            more, _ = story.place(box + (50, 50, -50, -50))
            story.draw(dev)
            writer.end_page()
        writer.close()
        with pymupdf.open(part) as one:
            out.insert_pdf(one)
    path = tmp_path / name
    out.save(path)
    out.close()
    return path


def search_hit(text, topic="t", source="a.md", modality="text", n=0, **meta):
    """A retrieved hit as the merger would hand it over."""
    from src.query.hybrid_search import SearchHit

    md = Metadata(topic, source, modality, **meta).to_dict()
    return SearchHit(id=f"{topic}/{source}/s{n}/0", topic_id=topic, text=text, metadata=md, score=0.03)
