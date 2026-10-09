import io

import pytest
from PIL import Image

from src import config
from src.ingestion.chunker import chunk_records
from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.image_parser import parse_image
from src.core.images import prepare_image
from src.ingestion.vision import ImageAnalysis, analysis_to_text


@pytest.fixture(autouse=True)
def isolated_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(config, "ROOT", tmp_path)


class FakeVision:
    def __init__(self, **fields):
        self.calls = 0
        self.fields = {"image_type": "document", "caption": "A shop receipt.", "text": "TOTAL 12.50",
                       "details": "", **fields}

    def __call__(self, data, mime):
        self.calls += 1
        return ImageAnalysis(**self.fields)


def make_image(tmp_path, name="r.jpg", size=(120, 80), mode="RGB", subdir=""):
    d = tmp_path / "data" / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    Image.new(mode, size, "white").save(p)
    rel = f"{subdir}/{name}" if subdir else name
    return RoutedFile(p, rel, "image")


def test_record_has_caption_ocr_text_and_a_link_to_the_saved_copy(tmp_path):
    [rec] = parse_image(make_image(tmp_path), "receipts", analyzer=FakeVision())
    assert rec.metadata.modality == "image" and rec.metadata.source == "r.jpg"
    assert "A shop receipt." in rec.text and "Text in image:\nTOTAL 12.50" in rec.text
    assert rec.metadata.image_path == "assets/receipts/images/r.jpg"
    assert (tmp_path / rec.metadata.image_path).exists()
    assert rec.label == "image r.jpg"


def test_empty_sections_are_left_out():
    a = ImageAnalysis(image_type="photo", caption="A red shirt.", text="  ", details="")
    assert analysis_to_text(a, "x.jpg") == "Image x.jpg (photo).\nA red shirt."
    c = ImageAnalysis(image_type="chart", caption="Bar chart.", text="Sales", details="Q1 10, Q2 20")
    assert analysis_to_text(c, "c.png").endswith("Chart/diagram details: Q1 10, Q2 20")


def test_unchanged_image_is_served_from_cache_without_a_vision_call(tmp_path):
    f, vision = make_image(tmp_path), FakeVision()
    first = parse_image(f, "t", analyzer=vision)
    second = parse_image(f, "t", analyzer=vision)
    assert vision.calls == 1 and first[0].text == second[0].text


def test_changed_image_or_new_prompt_version_or_model_triggers_reanalysis(tmp_path, monkeypatch):
    f, vision = make_image(tmp_path), FakeVision()
    parse_image(f, "t", analyzer=vision)
    Image.new("RGB", (120, 80), "black").save(f.path)                     # the file's content changed
    parse_image(f, "t", analyzer=vision)
    assert vision.calls == 2
    from src.ingestion.vision import PROMPT_VERSION
    monkeypatch.setattr("src.ingestion.vision.PROMPT_VERSION", PROMPT_VERSION + 1)                 # the prompt was improved
    parse_image(f, "t", analyzer=vision)
    assert vision.calls == 3
    monkeypatch.setattr(config, "LLM_MODEL", "some-other-model")
    parse_image(f, "t", analyzer=vision)
    assert vision.calls == 4


def test_same_file_name_in_different_folders_gets_separate_assets(tmp_path):
    a = parse_image(make_image(tmp_path, "p.jpg", subdir="a"), "t", analyzer=FakeVision())[0]
    b = parse_image(make_image(tmp_path, "p.jpg", subdir="b"), "t", analyzer=FakeVision())[0]
    assert a.metadata.image_path != b.metadata.image_path
    assert (tmp_path / a.metadata.image_path).exists() and (tmp_path / b.metadata.image_path).exists()


def test_corrupt_and_tiny_images_raise_value_error(tmp_path):
    bad = tmp_path / "data" / "bad.jpg"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        parse_image(RoutedFile(bad, "bad.jpg", "image"), "t", analyzer=FakeVision())
    with pytest.raises(ValueError, match="too small"):
        parse_image(make_image(tmp_path, "tiny.png", size=(5, 5)), "t", analyzer=FakeVision())


def test_prepare_image_caps_size_and_keeps_transparency_as_png():
    big = io.BytesIO()
    Image.new("RGB", (5000, 1000), "white").save(big, format="PNG")
    data, mime = prepare_image(big.getvalue())
    assert mime == "image/jpeg" and max(Image.open(io.BytesIO(data)).size) == 2000
    rgba = io.BytesIO()
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(rgba, format="PNG")
    assert prepare_image(rgba.getvalue())[1] == "image/png"


def test_chunker_gives_images_distinct_stable_ids(tmp_path):
    recs = [parse_image(make_image(tmp_path, n), "t", analyzer=FakeVision())[0] for n in ("a.jpg", "b.jpg")]
    ids = [c.id for c in chunk_records(recs)]
    assert ids == ["t/a.jpg/ia/0", "t/b.jpg/ib/0"]
