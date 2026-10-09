import json

import pytest

from src.query.generator import REFUSAL, Answer
from src.core.llm import Usage, record_usage, track_usage
from src.query.observability import QueryLog, Trace, format_trace
from src.query.pipeline import ERROR_TEXT, RagPipeline
from src.query.query_rewriter import RewriteResult
from src.query.router import RouteDecision
from tests.helpers import search_hit


class Recorder:
    """Fakes for every stage, recording how they were called."""

    def __init__(self, topics=("a",), hits=None, fail=(), standalone=None, answer=None, embed_error=None):
        self.calls: list[str] = []
        self.topics, self.fail, self.standalone, self.embed_error = list(topics), set(fail), standalone, embed_error
        self.hits = hits if hits is not None else {
            t: [search_hit(f"text of {t} one", topic=t, source=f"{t}.md", n=1), search_hit(f"text of {t} two", topic=t, source=f"{t}.md", n=2)]
            for t in self.topics}
        self.answer = answer or Answer("The answer is 42 [1].", True, [1], Usage(100, 10, 5))
        self.searched, self.vectors, self.rewrite_args, self.router_q, self.gen_args = [], [], [], [], []

    # --- fakes
    def route(self, question):
        self.calls.append("route"); self.router_q.append(question)
        return RouteDecision(self.topics, "because")

    def rewrite(self, question, history):
        self.calls.append("rewrite"); self.rewrite_args.append((question, history))
        q = self.standalone or question
        return RewriteResult(q, q != question, bool(history))

    def embed(self, question):
        self.calls.append("embed")
        if self.embed_error:
            raise self.embed_error
        return [0.1, 0.2, 0.3]

    def search(self, topic, question, k=None, query_vector=None):
        self.calls.append(f"search:{topic}"); self.searched.append((topic, question, k, query_vector))
        if topic in self.fail:
            raise RuntimeError(f"index for {topic} is broken")
        return self.hits.get(topic, [])

    def generate(self, question, context, tables=None):
        self.calls.append("generate"); self.gen_args.append((question, context)); self.tables_seen = tables
        return self.answer

    def catalog(self, topic_ids):
        self.catalog_topics = list(topic_ids)
        return self.tables

    tables: list = []                                                   # what the fake catalog returns
    catalog_topics: list = []
    tables_seen = None

    def pipeline(self, tmp_path=None, **kw):
        log = QueryLog(tmp_path / "q.jsonl") if tmp_path else QueryLog("/nonexistent-dir-for-tests/q.jsonl")
        return RagPipeline(router=self, searcher=self, rewrite=self.rewrite, embed=self.embed, generate=self.generate,
                           catalog=self.catalog, query_log=log, **kw)


# ------------------------------------------------------------------ the normal path
def test_all_stages_run_in_order_and_only_used_chunks_are_cited(tmp_path):
    r = Recorder(topics=("a", "b"))
    res = r.pipeline(tmp_path).answer("What is the answer?")
    assert r.calls == ["rewrite", "route", "embed", "search:a", "search:b", "generate"]
    assert [s.name for s in res.trace.steps] == ["rewrite", "route", "embed", "search", "merge", "context", "generate", "citations"]
    assert res.text == "The answer is 42 [1]." and res.answerable and res.refusal_stage is None and res.error is None
    assert len(res.context.items) == 4                                  # 2 topics x 2 chunks reached the generator...
    assert [s.numbers for s in res.sources] == [[1]]                    # ...but only chunk 1 is cited


def test_the_answer_shown_to_the_user_has_no_dangling_markers():
    r = Recorder(answer=Answer("The answer is 42 [1][7] [C3].", True, [1], Usage(10, 1, 0)))
    res = r.pipeline().answer("q")
    assert res.text == "The answer is 42 [1]." and [s.numbers for s in res.sources] == [[1]]


def test_the_question_is_embedded_once_and_the_same_vector_searches_every_topic():
    r = Recorder(topics=("a", "b", "c"))
    r.pipeline().answer("q")
    assert r.calls.count("embed") == 1
    assert [s[3] for s in r.searched] == [[0.1, 0.2, 0.3]] * 3
    assert {s[2] for s in r.searched} == {6}                            # config.TOP_K


def test_follow_up_is_searched_and_answered_with_the_standalone_question():
    r = Recorder(standalone="How many snakes were relocated in Mombasa?")
    hist = [{"role": "user", "content": "Nairobi?"}, {"role": "assistant", "content": "5"}]
    res = r.pipeline().answer("and in Mombasa?", hist)
    assert r.rewrite_args == [("and in Mombasa?", hist)]
    assert r.router_q == [r.standalone] and {s[1] for s in r.searched} == {r.standalone}
    assert r.gen_args[0][0] == r.standalone
    assert res.question == "and in Mombasa?" and res.standalone_question == r.standalone


def test_the_tables_of_the_routed_topics_are_offered_to_the_generator_for_calculations():
    from src.query.table_query import TableInfo
    r = Recorder(topics=("a", "b"))
    r.tables = [TableInfo("T1", "a", "x.csv", "main", 10, {"c": "integer"})]
    res = r.pipeline().answer("average of c?")
    assert r.catalog_topics == ["a", "b"] and r.tables_seen == r.tables
    assert res.trace.get("generate").info["tables"] == ["T1"]


# ------------------------------------------------------------------ the ways to refuse
def test_router_refusal_stops_before_any_search_or_generation():
    r = Recorder(topics=())
    res = r.pipeline().answer("recipe for pizza?")
    assert r.calls == ["rewrite", "route"]
    assert res.text == REFUSAL and not res.answerable and res.refusal_stage == "router" and res.sources == []
    assert [s.name for s in res.trace.steps] == ["rewrite", "route"]


def test_nothing_found_refuses_without_calling_the_generator():
    r = Recorder(topics=("a", "b"), hits={"a": [], "b": []})
    res = r.pipeline().answer("q")
    assert "generate" not in r.calls and res.refusal_stage == "retrieval" and res.text == REFUSAL


def test_generator_refusal_keeps_its_explanation_and_cites_nothing():
    r = Recorder(answer=Answer(f"{REFUSAL} Only receipts here.", False, [], Usage(50, 5, 0), "unsupported"))
    res = r.pipeline().answer("capital of France?")
    assert res.refusal_stage == "generator" and res.text.endswith("Only receipts here.") and res.sources == []
    assert res.trace.get("generate").info["refusal_reason"] == "unsupported"


# ------------------------------------------------------------------ failures never reach the user as tracebacks
def test_one_failing_topic_does_not_sink_the_others():
    r = Recorder(topics=("a", "b"), fail=("b",))
    res = r.pipeline().answer("q")
    assert res.answerable and len(res.context.items) == 2
    search = res.trace.get("search").info
    assert list(search["errors"]) == ["b"] and "index for b is broken" in search["errors"]["b"]


def test_every_topic_failing_gives_a_plain_error_message_not_an_exception():
    r = Recorder(topics=("a", "b"), fail=("a", "b"))
    res = r.pipeline().answer("q")
    assert res.text == ERROR_TEXT and res.refusal_stage == "error" and "search failed for every topic" in res.error
    assert res.trace.get("search").info["error"].startswith("RuntimeError")


def test_unexpected_exception_in_a_stage_is_contained_and_traced():
    r = Recorder(embed_error=ConnectionError("network down"))
    res = r.pipeline().answer("q")
    assert res.text == ERROR_TEXT and res.refusal_stage == "error" and "network down" in res.error
    assert "ConnectionError" in res.trace.get("embed").info["error"] and "generate" not in r.calls


@pytest.mark.parametrize("question,expected", [("", "Please type a question."), ("   ", "Please type a question."),
                                               ("x" * 2001, "too long")])
def test_bad_input_is_rejected_before_any_stage_runs(question, expected):
    r = Recorder()
    res = r.pipeline().answer(question)
    assert r.calls == [] and expected in res.text and res.refusal_stage == "input"


# ------------------------------------------------------------------ tracing and logging
def test_tokens_and_llm_calls_are_totalled_across_stages():
    r = Recorder()
    orig_rewrite, orig_generate = r.rewrite, r.generate
    r.rewrite = lambda q, h: (record_usage(Usage(10, 2, 0)), orig_rewrite(q, h))[1]
    r.generate = lambda q, c, tables=None: (record_usage(Usage(300, 40, 20)), orig_generate(q, c, tables))[1]
    res = r.pipeline().answer("q", [{"role": "user", "content": "hi"}])
    assert res.trace.llm_calls == 2 and res.trace.tokens == {"prompt": 310, "output": 42, "thinking": 20}
    assert res.trace.total_seconds > 0


def test_one_json_line_is_logged_per_question(tmp_path):
    r = Recorder(topics=("a", "b"))
    p = r.pipeline(tmp_path)
    p.answer("first question")
    p.answer("second question")
    lines = (tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert len(lines) == 2 and rec["question"] == "first question" and rec["answerable"] is True
    assert rec["sources"] == ["a · a.md · text"]
    assert [s["name"] for s in rec["trace"]["steps"]][:3] == ["rewrite", "route", "embed"]
    assert rec["trace"]["steps"][3]["per_topic"]["a"][0]["id"] == "a/a.md/s1/0"


def test_a_log_that_cannot_be_written_never_breaks_the_answer(tmp_path):
    r = Recorder()
    p = r.pipeline()
    p.query_log = QueryLog(tmp_path)                                     # a directory: opening it for append fails
    assert p.answer("q").answerable


# ------------------------------------------------------------------ the observability pieces on their own
def test_trace_step_times_records_info_and_marks_failures_then_reraises():
    t = Trace("q")
    with t.step("ok") as info:
        info["x"] = 1
    with pytest.raises(ValueError):
        with t.step("bad"):
            raise ValueError("nope")
    assert [s.name for s in t.steps] == ["ok", "bad"] and t.get("ok").info == {"x": 1}
    assert t.get("bad").info["error"] == "ValueError: nope" and t.get("missing") is None


def test_trace_dict_is_json_safe_and_the_text_form_names_every_step():
    t = Trace("q")
    with t.step("route") as info:
        info.update(topics=["a"], fallback=False, odd={1, 2})            # a set is not JSON by default
    t.finish([Usage(5, 1, 0)])
    json.dumps(t.to_dict())
    text = format_trace(t)
    assert "1 LLM call(s)" in text and "route" in text and "['a']" in text


def test_usage_is_only_collected_inside_a_tracking_scope():
    record_usage(Usage(1, 1, 1))                                         # outside any scope: ignored, no error
    with track_usage() as calls:
        record_usage(Usage(2, 3, 4))
    assert calls == [Usage(2, 3, 4)]
