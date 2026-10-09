from pathlib import Path

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import src.query.pipeline
from src import config
from src.query.citations import Source
from src.query.context import build_context
from src.query.generator import REFUSAL
from src.query.observability import Trace
from src.query.pipeline import PipelineResult
from tests.helpers import search_hit

APP = Path(__file__).resolve().parents[2] / "app.py"      # tests/interface/test_app.py -> project root


class FakePipeline:
    """Stands in for RagPipeline: records every call, returns scripted results."""

    def __init__(self, image_rel=None):
        self.calls = []
        self.image_rel = image_rel

    def answer(self, question, history=None):
        self.calls.append((question, [dict(h) for h in (history or [])]))
        trace = Trace(question)
        with trace.step("route") as info:
            info["topics"] = ["receipts"]
        trace.finish([])
        if "pizza" in question:
            return PipelineResult(question, question, REFUSAL, False, refusal_stage="router", trace=trace)
        hit = search_hit("TOTAL 18.00 on the receipt", topic="receipts", source="r.jpg", modality="image",
                         image_path=self.image_rel, n=1)
        ctx = build_context([hit])
        src = Source([1], "receipts", "r.jpg", "image", "", "receipts · r.jpg · image", image_path=self.image_rel)
        return PipelineResult(question, "standalone: " + question, f"The total is 18.00 [1]. (asked: {question})", True, [src],
                              context=ctx, retrieved=[hit], topics=["receipts"], trace=trace)


@pytest.fixture
def app(monkeypatch, tmp_path):
    st.cache_resource.clear()
    fake = FakePipeline()
    monkeypatch.setattr(src.query.pipeline, "RagPipeline", lambda: fake)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.fake = fake
    yield at
    st.cache_resource.clear()


def texts(at):
    return [m.value for m in at.markdown]


def ask(at, question):
    at.chat_input[0].set_value(question).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_the_page_loads_with_the_sidebar_examples_and_topics(app):
    app.run()
    assert not app.exception and app.header[0].value == "Ask your documents"
    assert len(app.sidebar.button) == 1 + 6                                     # "Clear" + the example questions
    assert any("Receipts" in e.label for e in app.sidebar.expander)               # the topics from topic.json are listed
    assert len(app.chat_message) == 0


def test_asking_shows_the_answer_the_sources_and_a_trace(app):
    app.run()
    ask(app, "What was the total?")
    roles = [m.name for m in app.chat_message]
    assert roles == ["user", "assistant"]
    assert app.chat_message[0].markdown[0].value == "What was the total?"
    assert "The total is 18.00 [1]." in app.chat_message[1].markdown[0].value
    assert any("receipts · r.jpg · image" in t for t in texts(app)) and any(t == "**Sources**" for t in texts(app))
    labels = [e.label for e in app.expander]
    assert any(l.startswith("Trace") for l in labels) and any(l == "Show the text of chunk [1]" for l in labels)
    assert any("Searched for:" in t and "standalone: What was the total?" in t for t in texts(app))


def test_the_conversation_history_reaches_the_pipeline_on_follow_ups(app):
    app.run()
    ask(app, "first question")
    ask(app, "and for the second?")
    assert app.fake.calls[0] == ("first question", [])
    q, hist = app.fake.calls[1]
    assert q == "and for the second?" and [h["role"] for h in hist] == ["user", "assistant"] and hist[0]["content"] == "first question"
    assert len(app.chat_message) == 4


def test_clicking_an_example_asks_it(app):
    app.run()
    app.sidebar.button(key="example0").click().run()
    assert not app.exception
    assert app.fake.calls and app.fake.calls[0][0].startswith("What was the total on receipt X51005361900")
    assert len(app.chat_message) == 2


def test_a_refusal_says_where_it_was_declined_and_lists_no_sources(app):
    app.run()
    ask(app, "recipe for pizza?")
    assert REFUSAL in app.chat_message[1].markdown[0].value
    assert any("declined at the **router** step" in c.value for c in app.caption)
    assert not any(t == "**Sources**" for t in texts(app))


def test_the_trace_can_be_hidden(app):
    app.run()
    ask(app, "q one")
    assert any(e.label.startswith("Trace") for e in app.expander)
    app.sidebar.toggle(key="show_trace").set_value(False).run()
    assert not any(e.label.startswith("Trace") for e in app.expander)


def test_clearing_the_conversation_empties_the_chat_and_the_memory(app):
    app.run()
    ask(app, "q one")
    clear = next(b for b in app.sidebar.button if b.label == "Clear the conversation")
    clear.click().run()
    assert len(app.chat_message) == 0
    ask(app, "after clearing")
    assert app.fake.calls[-1] == ("after clearing", [])


def test_a_source_picture_is_shown_when_the_file_exists(app, tmp_path):
    rel = "assets/receipts/images/r.jpg"
    (tmp_path / rel).parent.mkdir(parents=True)
    Image.new("RGB", (60, 80), "white").save(tmp_path / rel)
    app.fake.image_rel = rel
    app.run()
    ask(app, "show the receipt")
    assert len(app.get("image")) == 1                                             # the source's picture is on the page


def test_a_missing_source_picture_does_not_break_the_page(app):
    app.fake.image_rel = "assets/receipts/images/gone.jpg"
    app.run()
    ask(app, "show the receipt")
    assert not app.exception and len(app.get("image")) == 0
