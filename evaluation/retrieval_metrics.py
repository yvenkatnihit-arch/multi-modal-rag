"""Retrieval metrics: did search bring back the right evidence, and how much noise came with it?

Each test question labels the sources that SHOULD be retrieved (a file, optionally a page). The retriever returns chunks.

    recall    = relevant sources found / relevant sources            (did we miss any evidence?)
    precision = retrieved chunks that came from a relevant source
                / retrieved chunks                                    (how much of what we fetched is useful?)

Reading them: with k = 6 chunks and a single relevant file, precision cannot exceed (chunks from that file) / 6, so
precision is a measure of NOISE, not of success. Recall is the one that says whether retrieval works.

A relevant source "matches" a retrieved chunk when topic and file are equal and, if the label names a page, the page is
equal too. Questions that should be refused have no relevant sources: they score None and are left out of the means.
"""
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Ref:
    """A source label (on a relevant source) or a retrieved chunk's origin.
    `contains` (labels only): the chunk must also contain this text, so "some row of the catalog" does not count as
    finding the row for product 12839. `text` (retrieved chunks only): the chunk's text, compared against `contains`."""
    topic: str
    source: str
    page: int | None = None
    contains: str | None = None
    text: str = field(default="", compare=False, repr=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Ref":
        return cls(d["topic"], d["source"], d.get("page"), d.get("contains"))

    def __str__(self) -> str:
        return (f"{self.topic}/{self.source}" + (f" p.{self.page}" if self.page else "")
                + (f' containing "{self.contains}"' if self.contains else ""))


def matches(retrieved: Ref, relevant: Ref) -> bool:
    return (retrieved.topic == relevant.topic and retrieved.source == relevant.source
            and (relevant.page is None or retrieved.page == relevant.page)
            and (not relevant.contains or relevant.contains.casefold() in retrieved.text.casefold()))


@dataclass(frozen=True)
class RetrievalScore:
    precision: float | None          # None when the question has no relevant sources (nothing to measure against)
    recall: float | None
    retrieved: int                   # chunks considered
    relevant_chunks: int             # of those, how many came from a relevant source
    relevant_total: int
    relevant_found: int
    missed: list[Ref] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"precision": self.precision, "recall": self.recall, "retrieved": self.retrieved,
                "relevant_chunks": self.relevant_chunks, "relevant_total": self.relevant_total,
                "relevant_found": self.relevant_found, "missed": [str(m) for m in self.missed]}


def score_retrieval(retrieved: Sequence[Ref], relevant: Sequence[Ref], k: int | None = None) -> RetrievalScore:
    """Score the first k retrieved chunks (all of them when k is None) against the relevant sources."""
    considered = list(retrieved if k is None else retrieved[:k])
    if not relevant:
        return RetrievalScore(None, None, len(considered), 0, 0, 0)

    relevant_chunks = sum(any(matches(r, want) for want in relevant) for r in considered)
    found = [want for want in relevant if any(matches(r, want) for r in considered)]
    missed = [want for want in relevant if want not in found]
    return RetrievalScore(
        precision=relevant_chunks / len(considered) if considered else 0.0,
        recall=len(found) / len(relevant),
        retrieved=len(considered), relevant_chunks=relevant_chunks,
        relevant_total=len(relevant), relevant_found=len(found), missed=missed,
    )


def mean(values: Iterable[float | None]) -> float | None:
    """Average of the values that exist; None when there are none."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(scores: Iterable[RetrievalScore]) -> dict:
    scores = [s for s in scores if s.recall is not None]
    return {
        "questions": len(scores),
        "precision": mean(s.precision for s in scores),
        "recall": mean(s.recall for s in scores),
        "full_recall": sum(s.recall == 1.0 for s in scores),          # questions where every relevant source was found
        "no_recall": sum(s.recall == 0.0 for s in scores),            # questions where none was
    }
