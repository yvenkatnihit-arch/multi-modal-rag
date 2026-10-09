import io

import pymupdf
from PIL import Image

from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.pdf_parser import find_scanned_pages, parse_pdf, strip_inline_citations

BODY = pymupdf.Rect(50, 50, 550, 700)


def make_pdf(tmp_path, pages: list[list[str]], name="t.pdf"):
    """Each page is a list of paragraphs; every paragraph becomes its own text box (= one block)."""
    doc = pymupdf.open()
    for paras in pages:
        page = doc.new_page()
        y = 60
        for p in paras:
            page.insert_textbox(pymupdf.Rect(50, y, 550, y + 80), p, fontsize=10)
            y += 90
    path = tmp_path / name
    doc.save(path)
    return RoutedFile(path, name, "pdf")


def test_text_per_page_with_page_numbers(tmp_path):
    r = make_pdf(tmp_path, [["Alpha paragraph one is here."], ["Beta paragraph two is here."]])
    recs = parse_pdf(r, "t")
    assert [x.metadata.page for x in recs] == [1, 2]
    assert "Alpha" in recs[0].text and "Beta" in recs[1].text
    assert all(x.metadata.modality == "text" for x in recs)


def test_hyphenated_line_breaks_are_joined(tmp_path):
    r = make_pdf(tmp_path, [["The environ-\nment is changing fast for all of us."]])
    assert "environment is changing" in parse_pdf(r, "t")[0].text


def test_running_headers_and_page_numbers_removed(tmp_path):
    topics = ["Rainfall totals were above average.", "Sightings clustered near the river.",
              "Relocations peaked during March.", "Budget approval came in April.", "Staff training finished in May."]
    pages = [["ACME Annual Report", t, f"Page {i}"] for i, t in enumerate(topics, 1)]
    text = " ".join(x.text for x in parse_pdf(make_pdf(tmp_path, pages), "t"))
    assert "ACME Annual Report" not in text and "Page 3" not in text
    assert "Relocations peaked during March." in text


def test_references_section_removed_when_late(tmp_path):
    pages = [["Intro text that is long enough to count as content."],
             ["Main findings are discussed here in some detail."],
             ["Conclusion paragraph with final remarks.", "References", "[1] Smith J. A study. 2020.", "[2] Lee K. Another."]]
    text = " ".join(x.text for x in parse_pdf(make_pdf(tmp_path, pages), "t"))
    assert "Conclusion paragraph" in text and "Smith J" not in text and "Another" not in text


def test_references_word_early_in_document_is_kept(tmp_path):
    pages = [["Contents", "References", "Methods and the rest of the long document body text here."],
             ["More body text in the middle of the document with lots of words."],
             ["Even more body text near the end of the document, plenty of words."]]
    text = " ".join(x.text for x in parse_pdf(make_pdf(tmp_path, pages), "t"))
    assert "Methods and the rest" in text and "Even more body text" in text


def test_appendix_after_references_is_kept(tmp_path):
    pages = [["Body text of the paper with enough words in it to be long."],
             ["Closing remarks of the paper.", "References", "[1] Old source.", "Appendix A", "Extra table notes here."]]
    text = " ".join(x.text for x in parse_pdf(make_pdf(tmp_path, pages), "t"))
    assert "Old source" not in text and "Extra table notes" in text


def test_inline_citations_stripped():
    s = "Snakes are shy [3] and often [1, 2] misjudged (Smith et al., 2020) in cities (Lee and Kim 2019a)."
    assert strip_inline_citations(s) == "Snakes are shy and often misjudged in cities."
    assert "(2025)" in strip_inline_citations("The survey (2025) found 259 sightings.")   # lone year kept


def test_scanned_page_detected_and_skipped(tmp_path):
    buf = io.BytesIO()
    Image.new("L", (200, 100), 200).save(buf, format="PNG")
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(50, 50, 400, 250), stream=buf.getvalue())
    p = tmp_path / "scan.pdf"
    doc.save(p)
    r = RoutedFile(p, "scan.pdf", "pdf")
    assert find_scanned_pages(p) == [1]
    assert parse_pdf(r, "t", ocr=False) == []                    # with OCR off a scan is detected and skipped


def test_generated_survey_pdf_if_present():
    from src import config
    p = config.DATA_DIR / "wildlife_reports" / "urban_wildlife_survey_2025.pdf"
    if not p.exists():
        return
    recs = parse_pdf(RoutedFile(p, p.name, "pdf"), "wildlife_reports", ocr=False, tables=False, images=False)
    assert recs and "five Kenyan cities" in recs[0].text      # ligature 'ﬁve' repaired
