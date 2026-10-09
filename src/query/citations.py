"""Citations: the sources of an answer = ONLY what the answer actually used (known issue #6).

Used chunks       = numbers the model declared in used_chunks  UNION  the [n] markers in its answer text.
Used calculations = numbers declared in used_calculations      UNION  the [Cn] markers in its answer text.
Both are restricted to things that really exist (a made-up [9] is dropped and reported); a calculation must also
have succeeded. Several chunks from the same place collapse into one source that lists all their numbers.
A refusal cites nothing.
"""
import re
from dataclasses import dataclass, field

from src.query.context import Context
from src.query.generator import Answer
from src.core.records import Metadata, pdf_table_ref

_CHUNK_MARKER = re.compile(r"\[(\d+)\]")
_CALC_MARKER = re.compile(r"\[C(\d+)\]")


@dataclass(frozen=True)
class Source:
    numbers: list[int]             # context chunk numbers behind this source
    topic_id: str
    source: str                    # file, relative to the topic folder
    modality: str                  # text | table | image
    location: str                  # "p.3", "section: A > B", "sample rows 51-68", "summary", ... or ""
    display: str                   # "restaurants · zomato_restaurants.csv · sample rows 51-68 · table"
    page: int | None = None
    table_id: str | None = None
    image_path: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    calc_refs: list[str] = field(default_factory=list)     # ["C1"] for a calculation source
    calculation: str | None = None                         # what was computed, in words

    @property
    def refs(self) -> list[str]:
        """Everything in the answer text that points at this source: ['1', '4'] or ['C1']."""
        return [str(n) for n in self.numbers] + self.calc_refs


@dataclass(frozen=True)
class Citations:
    sources: list[Source]
    invalid: list[int] = field(default_factory=list)                # chunk numbers cited that are not in the context
    invalid_calculations: list[int] = field(default_factory=list)   # calculation numbers cited that do not exist / failed
    uncited: bool = False                                           # answered, yet cited nothing: a warning sign
    text: str = ""                                                  # the answer with markers that point at nothing removed


def strip_dangling_markers(text: str, valid_chunks: set[int], valid_calcs: set[int]) -> str:
    """Remove [n] / [Cn] markers that do not refer to a real chunk or a successful calculation (with the space before)."""
    text = re.sub(r"(\s?)\[(\d+)\]", lambda m: m.group(0) if int(m.group(2)) in valid_chunks else "", text)
    return re.sub(r"(\s?)\[C(\d+)\]", lambda m: m.group(0) if int(m.group(2)) in valid_calcs else "", text)


def _location(m: Metadata) -> str:
    if m.modality == "table":
        bits = []
        pdf = pdf_table_ref(m.table_id)
        if pdf:
            bits.append(f"p.{pdf[0]}, table {pdf[1]}")
        elif m.table_id and m.table_id != "main":
            bits.append(f"sheet {m.table_id}")
        if m.part:
            bits.append(m.part.replace("rows-", "rows ", 1).replace("sample-", "sample rows ", 1))
        return ", ".join(bits)
    if m.page:
        return f"p.{m.page}"
    return f"section: {m.heading}" if m.heading else ""


def build_citations(answer: Answer, context: Context) -> Citations:
    if not answer.answerable:
        return Citations([])

    declared = set(answer.used_chunks) | {int(n) for n in _CHUNK_MARKER.findall(answer.text)}
    valid = sorted(n for n in declared if context.item(n) is not None)
    invalid = sorted(declared - set(valid))

    grouped: dict[str, Source] = {}
    for n in valid:
        item = context.item(n)
        m = Metadata.from_dict(item.metadata)
        loc = _location(m)
        display = " · ".join(p for p in (m.topic_id, m.source, loc, m.modality) if p)
        if display in grouped:
            grouped[display].numbers.append(n)
            grouped[display].chunk_ids.append(item.chunk_id)
        else:
            grouped[display] = Source([n], m.topic_id, m.source, m.modality, loc, display,
                                      m.page, m.table_id, m.image_path, [item.chunk_id])
    sources = list(grouped.values())

    calc_declared = set(answer.used_calculations) | {int(n) for n in _CALC_MARKER.findall(answer.text)}
    by_number = {c.number: c for c in answer.calculations}
    valid_calc = sorted(n for n in calc_declared if n in by_number and by_number[n].ok)
    invalid_calc = sorted(calc_declared - set(valid_calc))
    for n in valid_calc:
        c = by_number[n]
        loc = f"{c.rows_matched:,} of {c.rows_total:,} rows"
        display = f"{c.topic_id} · {c.source} · calculation: {c.description} ({loc}) · table"
        sources.append(Source([], c.topic_id, c.source, "table", f"calculated over {loc}", display,
                              table_id=c.table_id, calc_refs=[c.ref], calculation=c.description))

    clean = strip_dangling_markers(answer.text, {n for n in range(1, len(context.items) + 1)}, set(valid_calc))
    return Citations(sources, invalid, invalid_calc, uncited=not sources, text=clean)
