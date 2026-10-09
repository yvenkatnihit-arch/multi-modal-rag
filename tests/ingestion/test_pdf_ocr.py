import io

import pymupdf
import pytest
from PIL import Image

from src import config
from src.ingestion.chunker import chunk_records
from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.pdf_parser import parse_pdf
from src.ingestion.vision import ImageAnalysis


@pytest.fixture(autouse=True)
def isolated_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(config, "ROOT", tmp_path)


class FakeVision:
    """Returns a different 'reading' per call so tests can tell pages apart; records the mime types it was sent."""

    def __init__(self, text="MEMO: 12 green snakes moved to Ngong", caption="A scanned field memo.", fail_on_call=None):
        self.calls, self.mimes, self.text, self.caption, self.fail_on_call = 0, [], text, caption, fail_on_call

    def __call__(self, data, mime):
        self.calls += 1
        self.mimes.append(mime)
        if self.fail_on_call == self.calls:
            raise RuntimeError("vision API down")
        return ImageAnalysis(image_type="document", caption=self.caption, text=f"{self.text} #{self.calls}", details="")


def scan_page(doc, seed=0):
    """A page that is only a picture, like a scanner produces: no text layer."""
    buf = io.BytesIO()
    Image.effect_noise((300, 200), 40 + seed).convert("L").save(buf, format="PNG")
    doc.new_page().insert_image(pymupdf.Rect(40, 40, 500, 340), stream=buf.getvalue())


def text_page(doc, text):
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 200), text, fontsize=11)


def make(tmp_path, kinds, name="mixed.pdf"):
    """kinds: 'scan' or a string of real text, one per page."""
    doc = pymupdf.open()
    for i, k in enumerate(kinds):
        scan_page(doc, i) if k == "scan" else text_page(doc, k)
    path = tmp_path / name
    doc.save(path)
    return RoutedFile(path, name, "pdf")


def test_a_scanned_page_becomes_a_text_record_with_its_page_number_and_image(tmp_path):
    vision = FakeVision()
    [rec] = parse_pdf(make(tmp_path, ["scan"], "memo.pdf"), "wildlife", analyzer=vision)
    assert rec.metadata.modality == "text" and rec.metadata.page == 1 and rec.label == "text p.1"
    assert rec.text.startswith("Scanned page 1 of memo.pdf (document).") and "A scanned field memo." in rec.text
    assert "Text on page:\nMEMO: 12 green snakes moved to Ngong #1" in rec.text
    assert rec.metadata.image_path == "assets/wildlife/pages/memo.pdf.p1.jpg" and (tmp_path / rec.metadata.image_path).exists()
    assert vision.mimes == ["image/jpeg"] and vision.calls == 1


def test_text_pages_and_scanned_pages_come_out_in_page_order_and_only_scans_are_read_by_vision(tmp_path):
    vision = FakeVision()
    recs = parse_pdf(make(tmp_path, ["The annual report opens with a long enough sentence of real text.", "scan",
                                     "Final page, again with a real text layer that is long enough."], "mix.pdf"),
                     "t", analyzer=vision)
    assert [r.metadata.page for r in recs] == [1, 2, 3] and vision.calls == 1
    assert recs[0].metadata.image_path is None and "annual report" in recs[0].text      # a text page is untouched
    assert recs[1].metadata.image_path and recs[1].text.startswith("Scanned page 2 of mix.pdf")


def test_several_scanned_pages_are_all_read_and_keep_their_own_page_numbers(tmp_path):
    vision = FakeVision()
    recs = parse_pdf(make(tmp_path, ["scan", "scan", "scan"], "batch.pdf"), "receipts", analyzer=vision)
    assert [r.metadata.page for r in recs] == [1, 2, 3] and vision.calls == 3
    assert len({r.metadata.image_path for r in recs}) == 3                               # one saved image per page


def test_rendering_is_capped_so_a_huge_scan_does_not_become_a_150_megapixel_image():
    from src.ingestion.parsers.pdf_parser import _render
    from src.core.images import MAX_SIDE
    doc = pymupdf.open()
    doc.new_page(width=2400, height=7000)                         # a tall receipt scan: 7000 points high
    doc.new_page(width=595, height=842)                           # an ordinary A4 page
    for page in doc:
        w, h = Image.open(io.BytesIO(_render(page))).size
        assert max(w, h) <= MAX_SIDE
        assert abs(w / h - page.rect.width / page.rect.height) < 0.01          # the shape is preserved
    small = pymupdf.open()
    small.new_page(width=200, height=300)
    assert abs(max(Image.open(io.BytesIO(_render(small[0]))).size) - 300 * 200 / 72) <= 1   # small pages: normal dpi, not blown up


def test_a_second_run_is_served_from_the_cache_without_calling_vision(tmp_path):
    f, vision = make(tmp_path, ["scan", "scan"], "again.pdf"), FakeVision()
    first = parse_pdf(f, "t", analyzer=vision)
    second = parse_pdf(f, "t", analyzer=vision)
    assert vision.calls == 2 and [r.text for r in first] == [r.text for r in second]


def test_one_failing_page_fails_the_whole_file_so_nothing_half_read_is_indexed(tmp_path):
    f = make(tmp_path, ["scan", "scan", "scan"], "flaky.pdf")
    with pytest.raises(RuntimeError, match="vision API down"):
        parse_pdf(f, "t", analyzer=FakeVision(fail_on_call=2))
    retry = FakeVision()
    assert len(parse_pdf(f, "t", analyzer=retry)) == 3                                    # the retry works
    assert 1 <= retry.calls <= 3          # pages read before the failure are cached; how many depends on thread timing


def test_ocr_can_be_switched_off_and_a_blank_reading_is_skipped(tmp_path):
    f = make(tmp_path, ["scan"], "off.pdf")
    never = FakeVision()
    assert parse_pdf(f, "t", ocr=False, analyzer=never) == [] and never.calls == 0
    nothing_readable = lambda data, mime: ImageAnalysis(image_type="other", caption="  ", text="", details="")  # noqa: E731
    assert parse_pdf(make(tmp_path, ["scan"], "blank.pdf"), "t", analyzer=nothing_readable) == []


def test_scanned_pages_get_distinct_stable_chunk_ids(tmp_path):
    recs = parse_pdf(make(tmp_path, ["scan", "scan"], "ids.pdf"), "t", analyzer=FakeVision())
    ids = [c.id for c in chunk_records(recs)]
    assert ids == ["t/ids.pdf/p1-iids.pdf.p1/0", "t/ids.pdf/p2-iids.pdf.p2/0"]
