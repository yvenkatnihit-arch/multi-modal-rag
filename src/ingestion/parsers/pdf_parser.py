"""PDF parser: one text Record per page.

Pages with a text layer: reading order, hyphenated line breaks, repeated headers/footers, the references
section and inline citations are all handled.
Pages WITHOUT a text layer (scans): rendered to an image and read by Gemini vision (OCR). The Record is ordinary
text for that page, carrying an image_path to the saved page image. Tables and embedded images come in Phase 5.
"""
import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import pymupdf

from src import config
from src.core.assets import page_asset_path, pdf_image_asset_path, relative_to_root
from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.common import clean_text
from src.ingestion.parsers.table_parser import table_records
from src.core.records import Metadata, Record
from src.core.images import MAX_SIDE
from src.ingestion.vision import ImageAnalysis, analysis_to_text, analyze_image, analyze_with_cache

log = logging.getLogger(__name__)

OCR_DPI = 200                # resolution a scanned page is rendered at before it is read
TABLE_OVERLAP = 0.6          # a text block this much inside a table's rectangle belongs to the table
MIN_TABLE_COLS = 2           # fewer columns than this is not a table
MIN_TABLE_ROWS = 1           # data rows, not counting the header
MIN_FILLED = 0.5             # a "table" with more than half its cells empty is a layout box, not data

# which embedded pictures count as figures worth reading
PDF_IMAGE_MIN_SIDE = 80              # pixels: smaller than this on either side is an icon or a bullet
PDF_IMAGE_MIN_AREA = 12_000          # pixels in total
PDF_IMAGE_MAX_ASPECT = 6             # longer / shorter side: beyond this it is a rule or a line
PDF_IMAGE_MIN_SHOWN = 40             # points: shown smaller than this on the page, it does not matter
PDF_IMAGE_PAGE_COVER = 0.9           # covering this much of the page = a background or watermark
PDF_IMAGE_DECORATION_SHARE = 0.5     # the same picture on this share of the pages (and at least 3) = a logo
MAX_PDF_IMAGES = 25                  # per PDF: bounds the vision cost; the largest are kept

MIN_TEXT_CHARS = 25          # fewer characters than this on a page = "no usable text layer"
REPEAT_MAX_LEN = 80          # only short blocks can be running headers/footers
REPEAT_SHARE = 0.5           # ...if they appear on at least half the pages
REPEAT_MIN_PAGES = 4         # ...in documents of at least this many pages
REFS_EARLIEST = 0.3          # a "References" heading in the first 30% of the text is not a real section

_REF_HEADING = re.compile(
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*|[ivx]+\.\s*)?(references|bibliography|works cited|literature cited)\s*:?\W*$",
    re.IGNORECASE,
)
_AFTER_REFS = re.compile(r"^\W*(?:[a-z0-9]+[.)]\s*)?(appendix|appendices|supplementary|supplement)\b", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
_NUM_CITATION = re.compile(r"\s?\[\d+(?:\s*[,–-]\s*\d+)*\]")
_AUTHOR_YEAR = re.compile(
    r"\s?\((?:[A-Z][A-Za-z'\-]+(?: et al\.?)?(?:,? (?:and|&) [A-Z][A-Za-z'\-]+)?,? (?:19|20)\d{2}[a-z]?(?:[;,] ?)?)+\)"
)


def _mostly_inside(block: pymupdf.Rect, areas: list[pymupdf.Rect]) -> bool:
    """True when at least TABLE_OVERLAP of the block lies inside one of the areas."""
    size = block.get_area()
    return size > 0 and any((block & a).get_area() / size >= TABLE_OVERLAP for a in areas)


def _page_blocks(page: pymupdf.Page, exclude: list[pymupdf.Rect] = ()) -> list[str]:
    """Text blocks of one page in reading order, each as a single cleaned paragraph.
    Blocks that lie inside an `exclude` area (a table, which is indexed separately) are left out."""
    out = []
    for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks", sort=True):
        if btype != 0:                       # 0 = text, 1 = image
            continue
        if exclude and _mostly_inside(pymupdf.Rect(x0, y0, x1, y1), list(exclude)):
            continue
        text = _HYPHEN_BREAK.sub(r"\1", text)
        text = clean_text(text.replace("\n", " "))
        if text:
            out.append(text)
    return out


def _repeated_keys(pages: list[list[str]]) -> set[str]:
    """Normalised short blocks that repeat on most pages: running headers, footers, page numbers."""
    if len(pages) < REPEAT_MIN_PAGES:
        return set()
    seen = Counter()
    for blocks in pages:
        seen.update({_key(b) for b in blocks if len(b) <= REPEAT_MAX_LEN})
    return {k for k, n in seen.items() if n >= REPEAT_SHARE * len(pages)}


def _key(block: str) -> str:
    return re.sub(r"\d+", "#", block.lower())


def _cut_references(pages: list[list[str]]) -> None:
    """Remove the references section in place (heading and everything after, up to an appendix)."""
    hits = [(pi, bi) for pi, blocks in enumerate(pages) for bi, b in enumerate(blocks) if _REF_HEADING.match(b)]
    if not hits:
        return
    pi, bi = hits[-1]                        # the last such heading is the real one
    chars = lambda blocks: sum(len(b) for b in blocks)
    total = sum(chars(b) for b in pages)
    before = sum(chars(b) for b in pages[:pi]) + chars(pages[pi][:bi])
    if before < total * REFS_EARLIEST:       # near the start: a contents entry, not a real section
        return
    for p in range(pi, len(pages)):
        blocks = pages[p]
        start = bi if p == pi else 0
        resume = next((i for i in range(start + 1, len(blocks)) if _AFTER_REFS.match(blocks[i])), None)
        if resume is None:
            pages[p] = blocks[:start]
        else:                                # an appendix follows: keep it
            pages[p] = blocks[:start] + blocks[resume:]


def strip_inline_citations(text: str) -> str:
    return _AUTHOR_YEAR.sub("", _NUM_CITATION.sub("", text))


def _is_scanned(page: pymupdf.Page, text: str) -> bool:
    return len(text.strip()) < MIN_TEXT_CHARS and bool(page.get_images(full=True))


def find_scanned_pages(path: Path) -> list[int]:
    """1-based numbers of pages that have images but no usable text layer (OCR candidates)."""
    with pymupdf.open(path) as doc:
        return [p.number + 1 for p in doc if _is_scanned(p, p.get_text())]


def _numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cells extracted from a PDF are all text ('34'). A column becomes numeric only if EVERY filled cell is a clean
    number, so a column of mixed content, or of codes like '007', stays text."""
    df = df.copy()
    for c in df.columns:
        s = df[c].dropna().astype(str).str.replace(",", "", regex=False).str.strip()
        if s.empty or not s.str.fullmatch(r"-?\d+(\.\d+)?").all():
            continue
        if s.str.fullmatch(r"-?0\d+").any():                    # leading zeros: an identifier, not a quantity
            continue
        df[c] = pd.to_numeric(df[c].astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")
    return df


def _restore_names(names: list[str]) -> list[str]:
    """PyMuPDF renames duplicate headers by prefixing every one with its position ('0-Name', '1-Name'). When that is
    clearly what happened (two or more columns reduce to the same name), give back the real name: 'Name', 'Name_2'."""
    stripped = [re.sub(r"^\d+-", "", n) for n in names]
    if len(set(stripped)) < len(stripped) and all(re.match(r"^\d+-", n) for n, s in zip(names, stripped) if stripped.count(s) > 1):
        names = [s if stripped.count(s) > 1 else n for n, s in zip(names, stripped)]
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 or not n else f"{n}_{seen[n]}")
    return out


def _tidy_table(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Clean what find_tables returned; None when it does not look like a real table."""
    df = raw.copy()
    df.columns = _restore_names([clean_text(str(c)).strip() if c is not None and not pd.isna(c) else "" for c in df.columns])
    df = df.map(lambda v: clean_text(str(v).replace("\n", " ")) or pd.NA if v is not None and not pd.isna(v) else pd.NA)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape[1] < MIN_TABLE_COLS or len(df) < MIN_TABLE_ROWS:
        return None
    if df.notna().to_numpy().mean() < MIN_FILLED:                # mostly empty cells: a layout box, not data
        return None
    df.columns = [c or f"column {i}" for i, c in enumerate(df.columns, 1)]
    return _numeric_columns(df)


def _find_tables(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, pd.DataFrame]]:
    """Real (ruled) tables on a text page, each with the rectangle it occupies."""
    try:
        found = page.find_tables().tables
    except Exception as e:                                       # a page the table finder cannot handle must not sink the file
        log.warning("table detection failed on page %d (%s: %s)", page.number + 1, type(e).__name__, e)
        return []
    out = []
    for t in found:
        try:
            df = _tidy_table(t.to_pandas())
        except Exception as e:
            log.warning("could not read a table on page %d (%s: %s)", page.number + 1, type(e).__name__, e)
            continue
        if df is not None:
            out.append((pymupdf.Rect(t.bbox), df))
    return out


@dataclass
class _Figure:
    page: int
    n: int                       # the n-th figure kept on its page
    data: bytes                  # JPEG, or PNG when the picture has transparency
    ext: str
    pixels: int
    digest: str


def _page_figures(doc: pymupdf.Document, page: pymupdf.Page) -> list[_Figure]:
    """Embedded pictures on one page that look like real figures, top to bottom."""
    page_area = page.rect.get_area()
    infos = [i for i in page.get_image_info(xrefs=True) if i.get("xref")]
    infos.sort(key=lambda i: (round(i["bbox"][1]), i["bbox"][0]))
    out, seen = [], set()
    for info in infos:
        xref, w, h, box = info["xref"], info["width"], info["height"], pymupdf.Rect(info["bbox"])
        if xref in seen:
            continue
        seen.add(xref)
        if min(w, h) < PDF_IMAGE_MIN_SIDE or w * h < PDF_IMAGE_MIN_AREA:
            continue                                                       # an icon or a bullet
        if max(w, h) / min(w, h) > PDF_IMAGE_MAX_ASPECT or min(box.width, box.height) < PDF_IMAGE_MIN_SHOWN:
            continue                                                       # a rule or a line, or shown too small to matter
        if page_area and box.get_area() / page_area >= PDF_IMAGE_PAGE_COVER:
            continue                                                       # a background or watermark behind the text
        try:
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:                                     # CMYK etc. -> RGB
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            data, ext = (pix.tobytes("png"), "png") if pix.alpha else (pix.tobytes("jpeg", jpg_quality=90), "jpg")
        except Exception as e:                                             # an unreadable image must not sink the PDF
            log.warning("could not extract image %d on page %d (%s: %s)", xref, page.number + 1, type(e).__name__, e)
            continue
        out.append(_Figure(page.number + 1, len(out) + 1, data, ext, pix.width * pix.height, hashlib.sha1(data).hexdigest()))
    return out


def _select_figures(found: list[_Figure], n_pages: int) -> list[_Figure]:
    """Drop decoration and repeats, cap the total, and number what is left per page."""
    pages_with: dict[str, set[int]] = defaultdict(set)
    for f in found:
        pages_with[f.digest].add(f.page)
    decoration_from = max(3, math.ceil(n_pages * PDF_IMAGE_DECORATION_SHARE))     # on half the pages = a logo
    kept, seen = [], set()
    for f in found:
        if len(pages_with[f.digest]) >= decoration_from or f.digest in seen:
            continue
        seen.add(f.digest)
        kept.append(f)
    if len(kept) > MAX_PDF_IMAGES:
        log.warning("%d figures found; reading only the %d largest", len(kept), MAX_PDF_IMAGES)
        biggest = {id(f) for f in sorted(kept, key=lambda f: -f.pixels)[:MAX_PDF_IMAGES]}
        kept = [f for f in kept if id(f) in biggest]
    counter: dict[int, int] = defaultdict(int)
    for f in kept:                                                         # gap-free numbering within each page
        counter[f.page] += 1
        f.n = counter[f.page]
    return kept


def _figure_records(figures: list[_Figure], routed: RoutedFile, topic_id: str, analyzer) -> list[Record]:
    def read(f: _Figure):
        asset = pdf_image_asset_path(topic_id, routed.source, f.page, f.n, f.ext)
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(f.data)                                          # the same bytes that are analysed
        return f, asset, analyze_with_cache(f.data, asset.with_name(asset.name + ".analysis.json"), analyzer)

    with ThreadPoolExecutor(max_workers=max(1, config.IMAGE_WORKERS)) as pool:
        done = list(pool.map(read, figures))                               # any failure fails the whole file

    name = routed.source.rsplit("/", 1)[-1]
    out = []
    for f, asset, a in done:
        if not (a.text.strip() or a.caption.strip() or a.details.strip()):
            continue
        text = analysis_to_text(a, f"{f.n} on page {f.page} of {name}")
        out.append(Record(text, Metadata(topic_id, routed.source, "image", page=f.page, image_path=relative_to_root(asset))))
    return out


def _render(page: pymupdf.Page) -> bytes:
    """The page as a JPEG: OCR_DPI, but never larger than the longest side the vision model is sent anyway. Without
    the cap a tall receipt scan renders at 150 megapixels, which is slow, memory-hungry and pointless."""
    zoom = min(OCR_DPI / 72, MAX_SIDE / max(page.rect.width, page.rect.height))
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("jpeg", jpg_quality=85)


def _ocr_pages(rendered: dict[int, bytes], routed: RoutedFile, topic_id: str, analyzer) -> dict[int, Record]:
    """Gemini vision reads each rendered scanned page. Any failure propagates, so the whole file fails and its old
    chunks are kept (pages that did succeed are cached, which makes the retry cheap)."""
    def read(n: int):
        asset = page_asset_path(topic_id, routed.source, n)
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(rendered[n])                                 # the page as an image, for the answer step
        analysis = analyze_with_cache(rendered[n], asset.with_name(asset.name + ".analysis.json"), analyzer)
        return n, asset, analysis

    with ThreadPoolExecutor(max_workers=max(1, config.IMAGE_WORKERS)) as pool:
        done = list(pool.map(read, sorted(rendered)))

    out: dict[int, Record] = {}
    for n, asset, a in done:
        if not (a.text.strip() or a.caption.strip() or a.details.strip()):
            log.warning("[%s] %s page %d: nothing readable on the scan; skipped", topic_id, routed.source, n)
            continue
        text = analysis_to_text(a, f"{n} of {routed.source}", noun="Scanned page", text_label="Text on page")
        out[n] = Record(text, Metadata(topic_id, routed.source, "text", page=n, image_path=relative_to_root(asset)))
    return out


def parse_pdf(
    routed: RoutedFile,
    topic_id: str,
    strip_citations: bool = True,
    ocr: bool = True,
    analyzer: Callable[[bytes, str], ImageAnalysis] = analyze_image,
    tables: bool = True,
    images: bool = True,
) -> list[Record]:
    pages: list[list[str]] = []
    scanned: list[int] = []
    page_tables: dict[int, list[tuple[pymupdf.Rect, pd.DataFrame]]] = {}
    figures: list[_Figure] = []
    with pymupdf.open(routed.path) as doc:
        if doc.needs_pass:
            raise ValueError(f"{routed.source} is password-protected")
        for p in doc:
            if _is_scanned(p, p.get_text()):
                scanned.append(p.number + 1)
                pages.append([])
                continue
            found = _find_tables(p) if tables else []
            if found:
                page_tables[p.number + 1] = found
            if images:
                figures += _page_figures(doc, p)
            pages.append(_page_blocks(p, [rect for rect, _ in found]))     # the table's own words are not indexed twice
        n_pages = len(doc)
        rendered = {n: _render(doc[n - 1]) for n in scanned} if ocr else {}     # PyMuPDF is not thread-safe: render here

    if scanned:
        (log.info if ocr else log.warning)("[%s] %s: pages %s have no text layer (scanned)%s", topic_id, routed.source,
                                           scanned, "; reading them with vision OCR" if ocr else "; OCR is off, skipped")

    repeated = _repeated_keys(pages)
    pages = [[b for b in blocks if _key(b) not in repeated] for blocks in pages]
    _cut_references(pages)

    records = []
    for number, blocks in enumerate(pages, 1):
        text = "\n\n".join(blocks)
        if strip_citations:
            text = strip_inline_citations(text)
        if len(text.strip()) < MIN_TEXT_CHARS:
            continue
        records.append(Record(text, Metadata(topic_id, routed.source, "text", page=number)))

    if rendered:
        by_page = {r.metadata.page: r for r in records}
        by_page.update(_ocr_pages(rendered, routed, topic_id, analyzer))
        records = [by_page[p] for p in sorted(by_page)]                # text pages and scanned pages, in page order

    for page_no, found in sorted(page_tables.items()):                 # every table becomes real table records + a saved copy
        for n, (_, df) in enumerate(found, 1):
            records += table_records(df, routed.source, topic_id, f"p{page_no}t{n}", page=page_no)

    if figures:                                                        # pictures inside the pages, read by vision
        records += _figure_records(_select_figures(figures, n_pages), routed, topic_id, analyzer)
    records.sort(key=lambda r: (r.metadata.page or 0, r.metadata.modality != "text"))     # by page; a page's text first
    return records
