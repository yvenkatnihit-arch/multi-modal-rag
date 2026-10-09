"""Context builder: the merger's ranked chunks  ->  one numbered, labelled text block for the generator.

    [1] text p.3 | wildlife_reports/urban_wildlife_survey_2025.pdf
    <chunk text>
    -----
    [2] table zomato_restaurants.csv (summary) | restaurants/zomato_restaurants.csv
    <chunk text>

Chunks are taken best-first until the token budget is full. A chunk that does not fit is skipped (a smaller,
lower-ranked one may still fit) and reported. The returned Context also keeps number -> metadata, which the
citations step needs later: the generator only ever sees the text.
"""
import math
from dataclasses import dataclass, field
from typing import Sequence

from src import config
from src.query.hybrid_search import SearchHit

SEPARATOR = "\n-----\n"
TRUNCATED = " …[truncated]"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / config.CHARS_PER_TOKEN)


def _neutralise(text: str) -> str:
    """Chunk text is untrusted data: it must not be able to close the <context>/<question> wrappers."""
    for tag in ("context", "question"):
        text = text.replace(f"</{tag}>", f"[/{tag}]").replace(f"<{tag}>", f"[{tag}]")
    return text


@dataclass(frozen=True)
class ContextItem:
    number: int                    # the [n] the generator cites
    chunk_id: str
    topic_id: str
    label: str                     # e.g. "text p.3", "table zomato.csv (summary)", "image fig2.png"
    metadata: dict                 # full chunk metadata (source, page, table_id, part, image_path...)
    text: str                      # what the generator actually sees (possibly truncated)
    tokens: int
    truncated: bool = False

    @property
    def image_path(self) -> str | None:
        return self.metadata.get("image_path")


@dataclass(frozen=True)
class Context:
    text: str
    items: list[ContextItem] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)      # chunk ids that did not fit the budget
    tokens: int = 0

    def __bool__(self) -> bool:
        return bool(self.items)

    def item(self, number: int) -> ContextItem | None:
        return self.items[number - 1] if 1 <= number <= len(self.items) else None


def build_context(hits: Sequence[SearchHit], max_tokens: int | None = None) -> Context:
    budget = max_tokens or config.CONTEXT_MAX_TOKENS
    items: list[ContextItem] = []
    dropped: list[str] = []
    used = 0

    for hit in hits:
        number = len(items) + 1
        header = f"[{number}] {hit.label} | {hit.topic_id}/{hit.metadata.get('source', '')}"
        body = _neutralise(hit.text)
        cost = estimate_tokens(header + "\n" + body) + estimate_tokens(SEPARATOR)
        truncated = False

        if used + cost > budget:
            if items:                                    # not the first: skip it, a smaller one may still fit
                dropped.append(hit.id)
                continue
            room_chars = max(int((budget - estimate_tokens(header)) * config.CHARS_PER_TOKEN) - len(TRUNCATED), 200)
            body, truncated = body[:room_chars] + TRUNCATED, True      # the top hit is never dropped outright
            cost = estimate_tokens(header + "\n" + body)

        items.append(ContextItem(number, hit.id, hit.topic_id, hit.label, hit.metadata, header + "\n" + body, cost, truncated))
        used += cost

    return Context(SEPARATOR.join(i.text for i in items), items, dropped, used)
