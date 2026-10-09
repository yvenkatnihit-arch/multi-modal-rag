"""Text parser: .txt .md .log .html .htm .docx  ->  one Record per section.

Every format is first reduced to a flat list of blocks:
    ("heading", level, text)   or   ("para", text)
and one shared routine groups the blocks into sections, tracking the heading path
("Risk > Mitigation"). Splitting long sections by size is the chunker's job, not ours.
"""
import re

from bs4 import BeautifulSoup
from docx import Document
from docx.table import Table

from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.common import clean_text, read_text_file
from src.core.records import Metadata, Record

Block = tuple  # ("heading", level, text) | ("para", text)

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HTML_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "tr"]


# ---------------------------------------------------------------- format -> blocks
def _markdown_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []
    in_fence = False

    def flush():
        if para:
            blocks.append(("para", "\n".join(para)))
            para.clear()

    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _MD_HEADING.match(line)
        if m:
            flush()
            blocks.append(("heading", len(m.group(1)), m.group(2)))
        elif not line.strip() and not in_fence:
            flush()
        else:
            para.append(line)
    flush()
    return blocks


def _html_blocks(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    root = soup.body or soup
    blocks: list[Block] = []
    for el in root.find_all(_HTML_BLOCKS):
        if el.find(_HTML_BLOCKS):          # container of other blocks: its children are visited instead
            continue
        if el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            text = " | ".join(c for c in cells if c)
        else:
            text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name[0] == "h" and el.name[1:].isdigit():
            blocks.append(("heading", int(el.name[1:]), text))
        else:
            blocks.append(("para", text))
    return blocks


def _docx_blocks(path) -> list[Block]:
    doc = Document(str(path))
    blocks: list[Block] = []
    for item in doc.iter_inner_content():      # paragraphs and tables in document order
        if isinstance(item, Table):
            for row in item.rows:
                cells: list[str] = []
                for c in row.cells:            # merged cells repeat; drop consecutive duplicates
                    t = c.text.strip()
                    if t and (not cells or cells[-1] != t):
                        cells.append(t)
                if cells:
                    blocks.append(("para", " | ".join(cells)))
            continue
        text = item.text.strip()
        if not text:
            continue
        style = (item.style.name or "") if item.style is not None else ""
        if style == "Title":
            blocks.append(("heading", 1, text))
        elif style.startswith("Heading ") and style[8:].isdigit():
            blocks.append(("heading", int(style[8:]) + 1, text))   # Title is level 1
        else:
            blocks.append(("para", text))
    return blocks


# ---------------------------------------------------------------- blocks -> sections
def _sections(blocks: list[Block]) -> list[tuple[str | None, str]]:
    """Group blocks into (heading_path, body) sections. Heading-only sections are dropped
    but their heading stays in the path of what follows."""
    out: list[tuple[str | None, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def path() -> str | None:
        return " > ".join(t for _, t in stack) or None

    def flush():
        text = clean_text("\n\n".join(body))
        if re.search(r"\w", text):       # drop empty sections and pure separators like '---'
            out.append((path(), text))
        body.clear()

    for b in blocks:
        if b[0] == "heading":
            flush()
            _, level, text = b
            text = clean_text(text)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        else:
            body.append(b[1])
    flush()
    return out


# ---------------------------------------------------------------- public entry point
def parse_text(routed: RoutedFile, topic_id: str) -> list[Record]:
    ext = routed.path.suffix.lower()
    if ext == ".docx":
        blocks = _docx_blocks(routed.path)
    elif ext in (".html", ".htm"):
        blocks = _html_blocks(read_text_file(routed.path))
    elif ext == ".json":                    # a JSON file the router judged to be text
        blocks = [("para", read_text_file(routed.path))]
    elif ext == ".log":                     # logs have no structure; '#' is not a heading there
        blocks = [("para", read_text_file(routed.path))]
    else:                                   # .txt .md
        blocks = _markdown_blocks(read_text_file(routed.path))

    return [
        Record(text, Metadata(topic_id, routed.source, "text", heading=heading))
        for heading, text in _sections(blocks)
    ]
