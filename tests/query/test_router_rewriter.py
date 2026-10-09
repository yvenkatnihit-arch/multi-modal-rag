import pytest

from src.query.query_rewriter import Rewrite, format_history, rewrite_question
from src.query.router import RouteChoice, TopicRouter
from src.core.topic_registry import Topic

TOPICS = [
    Topic("receipts", "Receipts", "Scanned shop receipts: totals, dates, shop names.", None),
    Topic("restaurants", "Restaurants", "Zomato restaurant table: cuisines, ratings, prices.", None),
    Topic("song_lyrics", "Song lyrics", "Lyrics of 25 songs.", None),
    Topic("wildlife_reports", "Wildlife reports", "Snake sightings and risk reports.", None),
]


class FakeLLM:
    """Records every call and returns a canned answer (or raises)."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def __call__(self, prompt, schema, system=None):
        self.calls.append((prompt, schema, system))
        if self.error:
            raise self.error
        return self.answer


# ------------------------------------------------------------------ router
def test_router_prompt_contains_every_topic_description_and_the_question():
    llm = FakeLLM(RouteChoice(topics=["receipts"], reason="r"))
    TopicRouter(TOPICS, llm).route("what was the total?")
    prompt, schema, system = llm.calls[0]
    assert schema is RouteChoice and "at most 3" in system
    for t in TOPICS:
        assert f"id: {t.id}" in prompt and t.description in prompt
    assert "<question>\nwhat was the total?\n</question>" in prompt


def test_router_rules_say_descriptions_are_not_exhaustive_field_lists():
    llm = FakeLLM(RouteChoice(topics=["restaurants"], reason="r"))
    TopicRouter(TOPICS, llm).route("what is the latitude of X?")
    system = llm.calls[0][2]
    assert "NOT complete lists of its fields" in system and "When you are unsure" in system


def test_router_returns_chosen_topics_in_order():
    d = TopicRouter(TOPICS, FakeLLM(RouteChoice(topics=["restaurants", "receipts"], reason="both"))).route("q")
    assert d.topics == ["restaurants", "receipts"] and d.reason == "both" and not d.fallback


def test_empty_list_means_refuse_and_is_not_a_fallback():
    d = TopicRouter(TOPICS, FakeLLM(RouteChoice(topics=[], reason="small talk"))).route("hi")
    assert d.topics == [] and not d.fallback


def test_unknown_duplicate_and_excess_topics_are_cleaned():
    llm = FakeLLM(RouteChoice(topics=["receipts", "made_up", "receipts", "restaurants", "song_lyrics", "wildlife_reports"], reason="x"))
    d = TopicRouter(TOPICS, llm, max_topics=3).route("q")
    assert d.topics == ["receipts", "restaurants", "song_lyrics"]
    assert d.dropped == ("made_up",)


def test_only_unknown_topics_falls_back_to_all():
    d = TopicRouter(TOPICS, FakeLLM(RouteChoice(topics=["nope"], reason="x"))).route("q")
    assert d.fallback and d.topics == [t.id for t in TOPICS]


@pytest.mark.parametrize("error", [RuntimeError("boom"), ValueError("blocked")])
def test_llm_failure_falls_back_to_all_topics_without_raising(error):
    d = TopicRouter(TOPICS, FakeLLM(error=error)).route("q")
    assert d.fallback and d.topics == [t.id for t in TOPICS] and "unavailable" in d.reason


def test_no_topics_means_nothing_to_search():
    llm = FakeLLM(RouteChoice(topics=["x"], reason="x"))
    assert TopicRouter([], llm).route("q").topics == [] and llm.calls == []


# ------------------------------------------------------------------ rewriter
HISTORY = [
    {"role": "user", "content": "How many snakes were relocated in Nairobi?"},
    {"role": "assistant", "content": "5 snakes were relocated in Nairobi [wildlife_reports, survey.pdf p.1]."},
]


def test_no_history_means_no_llm_call():
    llm = FakeLLM(Rewrite(standalone_question="SHOULD NOT BE USED"))
    for h in (None, []):
        r = rewrite_question("  How many receipts?  ", h, llm)
        assert r.question == "How many receipts?" and not r.changed and not r.used_llm
    assert llm.calls == []


def test_follow_up_is_rewritten_using_history():
    llm = FakeLLM(Rewrite(standalone_question="How many snakes were relocated in Mombasa?"))
    r = rewrite_question("and in Mombasa?", HISTORY, llm)
    assert r.question == "How many snakes were relocated in Mombasa?" and r.changed and r.used_llm
    prompt, _, system = llm.calls[0]
    assert "User: How many snakes" in prompt and "Assistant: 5 snakes" in prompt
    assert "<latest_question>\nand in Mombasa?\n</latest_question>" in prompt and "Do NOT answer" in system


def test_unchanged_question_is_reported_as_unchanged():
    r = rewrite_question("What is the average rating?", HISTORY, FakeLLM(Rewrite(standalone_question="What is the average rating?")))
    assert not r.changed and r.used_llm


@pytest.mark.parametrize("llm", [
    FakeLLM(error=RuntimeError("api down")),
    FakeLLM(Rewrite(standalone_question="   ")),
    FakeLLM(Rewrite(standalone_question="x" * 601)),
])
def test_bad_rewrite_keeps_the_original_question(llm):
    r = rewrite_question("and in Mombasa?", HISTORY, llm)
    assert r.question == "and in Mombasa?" and not r.changed and r.error


def test_history_is_limited_and_long_messages_are_cut():
    long = [{"role": "user", "content": f"message {i}"} for i in range(10)]
    text = format_history(long, max_messages=3)
    assert text.splitlines() == ["User: message 7", "User: message 8", "User: message 9"]
    cut = format_history([{"role": "assistant", "content": "word " * 500}], max_chars=50)
    assert len(cut) < 70 and cut.endswith("…")
