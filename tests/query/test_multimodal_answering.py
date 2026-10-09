from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import BaseModel

from src import config
from src.core import llm
from src.query.context import build_context
from src.query.generator import SYSTEM, GeneratedAnswer, generate_answer
from src.core.llm import Usage
from src.query.pictures import Picture, select_pictures
from tests.helpers import search_hit


@pytest.fixture(autouse=True)
def project_root_is_the_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)


def save(tmp_path, rel, size=(300, 200), data=None):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        path.write_bytes(data)
    else:
        Image.effect_noise(size, 50).convert("RGB").save(path)
    return rel


def image_hit(rel, n, source="x.jpg", topic="t"):
    return search_hit(f"description of {source}", topic=topic, source=source, modality="image", image_path=rel, n=n)


# ------------------------------------------------------------------ choosing and loading pictures
def test_pictures_are_loaded_best_first_one_per_file_and_capped(tmp_path):
    a, b, c, d = (save(tmp_path, f"assets/t/images/{n}.jpg") for n in "abcd")
    ctx = build_context([image_hit(a, 1, "a.jpg"), search_hit("plain text chunk", n=2), image_hit(a, 3, "a-again.jpg"),
                         image_hit(b, 4, "b.jpg"), image_hit(c, 5, "c.jpg"), image_hit(d, 6, "d.jpg")])
    pics = select_pictures(ctx)
    assert [p.number for p in pics] == [1, 4, 5]                                    # chunk 3 repeats a.jpg; d.jpg is over the cap
    assert [p.path for p in pics] == [a, b, c] and all(p.mime == "image/jpeg" and p.data for p in pics)
    assert pics[0].caption == "Picture for chunk [1] (image a.jpg):"
    assert [p.number for p in select_pictures(ctx, max_n=1)] == [1] and select_pictures(ctx, max_n=0) == []


def test_a_missing_or_corrupt_picture_is_skipped_without_failing(tmp_path):
    good = save(tmp_path, "assets/t/images/good.jpg")
    bad = save(tmp_path, "assets/t/images/bad.jpg", data=b"not an image at all")
    ctx = build_context([image_hit("assets/t/images/gone.jpg", 1, "gone.jpg"), image_hit(bad, 2, "bad.jpg"), image_hit(good, 3, "good.jpg")])
    assert [p.number for p in select_pictures(ctx)] == [3]


def test_chunks_without_a_picture_give_no_pictures():
    assert select_pictures(build_context([search_hit("just text", n=1)])) == []


def test_a_scanned_pdf_page_with_a_saved_page_image_counts_as_a_picture(tmp_path):
    rel = save(tmp_path, "assets/t/pages/memo.pdf.p1.jpg")
    ctx = build_context([search_hit("OCR text of the memo", source="memo.pdf", page=1, image_path=rel, n=1)])   # a TEXT chunk
    assert [p.number for p in select_pictures(ctx)] == [1]


# ------------------------------------------------------------------ the generator shows them to the model
class Script:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, prompt, schema, system=None, thinking_budget=0, **extra):
        self.calls.append({"prompt": prompt, "system": system, "extra": extra})
        return self.replies[min(len(self.calls), len(self.replies)) - 1], Usage(10, 1, 0)


def fake_pictures(*numbers):
    return lambda ctx: [Picture(n, "image x.jpg", f"p{n}.jpg", b"JPEGBYTES", "image/jpeg") for n in numbers]


CTX = build_context([search_hit("a description", modality="image", n=1, image_path="assets/t/images/a.jpg"), search_hit("more", n=2)])
ANSWER = GeneratedAnswer(answerable=True, answer="It is grey [1].", used_chunks=[1])


def test_pictures_are_sent_with_the_prompt_and_listed_in_it_and_recorded_in_the_answer():
    s = Script(ANSWER)
    a = generate_answer("what colour?", CTX, s, pick_pictures=fake_pictures(1))
    [call] = s.calls
    assert call["extra"]["images"] == [("Picture for chunk [1] (image x.jpg):", b"JPEGBYTES", "image/jpeg")]
    assert "<pictures>\nThe original pictures shown before this message belong to chunks: [1] (image x.jpg).\n</pictures>" in call["prompt"]
    assert a.pictures_shown == [1] and a.answerable


def test_without_pictures_nothing_extra_is_passed_and_the_prompt_has_no_picture_note():
    s = Script(ANSWER)
    a = generate_answer("q", CTX, s, pick_pictures=fake_pictures())
    assert s.calls[0]["extra"] == {} and "<pictures>" not in s.calls[0]["prompt"] and a.pictures_shown == []


def test_pictures_can_be_switched_off():
    s = Script(ANSWER)
    called = []
    a = generate_answer("q", CTX, s, pictures=False, pick_pictures=lambda c: called.append(1) or fake_pictures(1)(c))
    assert called == [] and s.calls[0]["extra"] == {} and a.pictures_shown == []


def test_pictures_accompany_every_round_of_a_calculation_loop():
    from src.query.table_query import Filter, TableInfo, TableRequest
    req = GeneratedAnswer(answerable=True, answer="", table_requests=[TableRequest(table="T1", operation="count")])
    s = Script(req, ANSWER)
    tables = [TableInfo("T1", "t", "x.csv", "main", 3, {"c": "integer"})]
    generate_answer("q", CTX, s, tables=tables, run_requests=lambda reqs, start: [], pick_pictures=fake_pictures(1))
    assert len(s.calls) == 2 and all(c["extra"].get("images") for c in s.calls)


def test_a_refusal_still_records_which_pictures_were_shown():
    s = Script(GeneratedAnswer(answerable=False, answer="Not shown in the picture."))
    a = generate_answer("q", CTX, s, pick_pictures=fake_pictures(1))
    assert not a.answerable and a.pictures_shown == [1]


def test_the_system_prompt_tells_the_model_to_trust_the_picture_over_its_description():
    assert "PICTURES" in SYSTEM and "trust the picture" in SYSTEM and "primary" in SYSTEM


def test_the_system_prompt_forbids_filling_in_the_unsupported_part_of_a_question():
    """Regression: asked for a photo's colour AND the catalog's colour, the model invented the catalog value."""
    assert "PARTLY ANSWERABLE" in SYSTEM and "state plainly which part could not be found" in SYSTEM
    assert "What a picture shows is not what a catalog, table or document says" in SYSTEM


# ------------------------------------------------------------------ what is really sent to Gemini
class Reply(BaseModel):
    ok: bool = True


class FakeGemini:
    def __init__(self):
        self.contents = None
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.contents = contents
        return SimpleNamespace(parsed=Reply(), text="{}", usage_metadata=SimpleNamespace(
            prompt_token_count=5, candidates_token_count=1, thoughts_token_count=0))


def test_the_request_to_gemini_puts_each_picture_after_its_caption_and_before_the_prompt(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    llm.generate_structured_ex("THE PROMPT", Reply, images=[("cap A", b"AAAA", "image/jpeg"), ("cap B", b"BBBB", "image/png")])
    parts = fake.contents
    assert [p if isinstance(p, str) else (p.inline_data.mime_type, p.inline_data.data) for p in parts] == [
        "cap A", ("image/jpeg", b"AAAA"), "cap B", ("image/png", b"BBBB"), "THE PROMPT"]


def test_without_pictures_the_request_is_just_the_prompt(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    assert llm.generate_structured_ex("ONLY TEXT", Reply)[1] == Usage(5, 1, 0)
    assert fake.contents == "ONLY TEXT"
