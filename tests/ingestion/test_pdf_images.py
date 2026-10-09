import io

import pymupdf
import pytest
from PIL import Image

from src import config
from src.ingestion.chunker import chunk_records
from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers import pdf_parser
from src.ingestion.parsers.pdf_parser import parse_pdf
from src.core.records import make_label
from src.ingestion.vision import ImageAnalysis
from tests.helpers import html_pdf


@pytest.fixture(autouse=True)
def project_root_is_the_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")


class FakeVision:
    def __init__(self, fail_on=None, empty=False):
        self.calls, self.mimes, self.fail_on, self.empty = 0, [], fail_on, empty

    def __call__(self, data, mime):
        self.calls += 1
        self.mimes.append(mime)
        if self.fail_on == self.calls:
            raise RuntimeError("vision API down")
        if self.empty:
            return ImageAnalysis(image_type="other", caption=" ", text="", details="")
        return ImageAnalysis(image_type="chart", caption=f"A bar chart (figure read #{self.calls}).", text="Sightings 34 55", details="")


def picture(tmp_path, name, size=(400, 260)):
    """A noisy image (it does not compress to nothing, and every call gives different pixels)."""
    Image.effect_noise(size, 60).convert("RGB").save(tmp_path / name)


def fig(name, width=300):
    return f'<p><img src="{name}" width="{width}"/></p>'


def make(tmp_path, bodies, name="doc.pdf"):
    return RoutedFile(html_pdf(tmp_path, bodies, name), name, "pdf")


TEXT = "<p>Ordinary prose that is long enough to count as the text of this page of the report.</p>"


def figures(recs):
    return [r for r in recs if r.metadata.modality == "image"]


def test_an_embedded_figure_becomes_an_image_record_with_its_page_and_a_saved_copy(tmp_path):
    picture(tmp_path, "chart.png")
    vision = FakeVision()
    recs = parse_pdf(make(tmp_path, [TEXT + fig("chart.png")], "survey.pdf"), "wild", ocr=False, analyzer=vision)
    [img] = figures(recs)
    assert img.metadata.page == 1 and img.metadata.source == "survey.pdf"
    assert img.text.startswith("Image 1 on page 1 of survey.pdf (chart).") and "A bar chart (figure read #1)." in img.text
    assert img.metadata.image_path == "assets/wild/pdf_images/survey.pdf.p1i1.jpg" and (tmp_path / img.metadata.image_path).exists()
    assert make_label(img.metadata) == "image survey.pdf p.1" and vision.mimes == ["image/jpeg"]
    assert any(r.metadata.modality == "text" for r in recs)                                  # the page text is still there


def test_icons_thin_rules_and_page_sized_backgrounds_are_not_read(tmp_path):
    picture(tmp_path, "icon.png", (40, 40))
    picture(tmp_path, "rule.png", (700, 20))
    vision = FakeVision()
    base = make(tmp_path, [TEXT + fig("icon.png", 30) + fig("rule.png", 400)], "small.pdf")
    assert figures(parse_pdf(base, "t", ocr=False, analyzer=vision)) == [] and vision.calls == 0

    doc = pymupdf.open(base.path)                                                             # now a full-page background image
    buf = io.BytesIO()
    Image.effect_noise((600, 800), 50).convert("RGB").save(buf, format="PNG")
    doc[0].insert_image(doc[0].rect, stream=buf.getvalue(), overlay=False)
    bg = tmp_path / "bg.pdf"
    doc.save(bg)
    doc.close()
    assert figures(parse_pdf(RoutedFile(bg, "bg.pdf", "pdf"), "t", ocr=False, analyzer=vision)) == [] and vision.calls == 0


def test_a_logo_on_every_page_is_ignored_and_a_repeated_figure_is_indexed_once(tmp_path):
    picture(tmp_path, "logo.png")
    picture(tmp_path, "figure.png")
    pages = [TEXT + fig("logo.png", 200) + (fig("figure.png") if i in (1, 3) else "") for i in range(4)]
    vision = FakeVision()
    recs = figures(parse_pdf(make(tmp_path, pages, "report.pdf"), "t", ocr=False, analyzer=vision))
    assert [r.metadata.page for r in recs] == [2] and vision.calls == 1                      # only the figure, first occurrence


def test_two_figures_on_one_page_are_numbered_top_to_bottom_with_distinct_ids(tmp_path):
    picture(tmp_path, "a.png")
    picture(tmp_path, "b.png", (500, 300))
    recs = parse_pdf(make(tmp_path, [TEXT + fig("a.png") + TEXT + fig("b.png", 250)], "two.pdf"), "t", ocr=False,
                     analyzer=FakeVision())
    imgs = figures(recs)
    assert [i.text.split(" (")[0] for i in imgs] == ["Image 1 on page 1 of two.pdf", "Image 2 on page 1 of two.pdf"]
    assert len({i.metadata.image_path for i in imgs}) == 2
    ids = [c.id for c in chunk_records(recs)]
    assert len(ids) == len(set(ids)) and any(i.endswith("/p1-itwo.pdf.p1i2/0") for i in ids)


def test_a_second_run_is_served_from_the_cache(tmp_path):
    picture(tmp_path, "chart.png")
    f, vision = make(tmp_path, [TEXT + fig("chart.png")]), FakeVision()
    first, second = parse_pdf(f, "t", ocr=False, analyzer=vision), parse_pdf(f, "t", ocr=False, analyzer=vision)
    assert vision.calls == 1 and [r.text for r in first] == [r.text for r in second]


def test_a_vision_failure_fails_the_whole_file(tmp_path):
    picture(tmp_path, "chart.png")
    with pytest.raises(RuntimeError, match="vision API down"):
        parse_pdf(make(tmp_path, [TEXT + fig("chart.png")]), "t", ocr=False, analyzer=FakeVision(fail_on=1))


def test_images_can_be_switched_off_and_unreadable_ones_are_skipped(tmp_path):
    picture(tmp_path, "chart.png")
    f = make(tmp_path, [TEXT + fig("chart.png")])
    off = FakeVision()
    assert figures(parse_pdf(f, "t", ocr=False, images=False, analyzer=off)) == [] and off.calls == 0
    assert figures(parse_pdf(f, "t", ocr=False, analyzer=FakeVision(empty=True))) == []


def test_only_the_largest_figures_are_read_when_there_are_too_many(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_parser, "MAX_PDF_IMAGES", 2)
    sizes = {"s1.png": (300, 200), "s2.png": (600, 400), "s3.png": (350, 220), "s4.png": (700, 450)}
    for n, s in sizes.items():
        picture(tmp_path, n, s)
    vision = FakeVision()
    pages = [TEXT + fig(n, 220) for n in sizes]
    recs = figures(parse_pdf(make(tmp_path, pages, "many.pdf"), "t", ocr=False, analyzer=vision))
    assert vision.calls == 2 and [r.metadata.page for r in recs] == [2, 4]                    # s2 and s4: the largest, in page order


def test_scanned_pages_are_read_once_by_ocr_and_not_again_as_figures(tmp_path):
    buf = io.BytesIO()
    Image.effect_noise((500, 400), 50).convert("L").save(buf, format="PNG")
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(40, 40, 500, 400), stream=buf.getvalue())
    path = tmp_path / "scan.pdf"
    doc.save(path)
    vision = FakeVision()
    recs = parse_pdf(RoutedFile(path, "scan.pdf", "pdf"), "t", analyzer=vision)
    assert vision.calls == 1 and [r.metadata.modality for r in recs] == ["text"]


def test_the_generated_survey_pdf_has_its_chart_extracted():
    p = config.DATA_DIR / "wildlife_reports" / "urban_wildlife_survey_2025.pdf"
    if not p.exists():
        pytest.skip("sample data not present")
    vision = FakeVision()
    recs = parse_pdf(RoutedFile(p, p.name, "pdf"), "wildlife_reports", ocr=False, analyzer=vision)
    [chart] = figures(recs)
    assert chart.metadata.page == 1 and vision.calls == 1
    assert {r.metadata.modality for r in recs} == {"text", "table", "image"}                  # all three kinds from one PDF
