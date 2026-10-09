import json
import re

import pandas as pd
import pytest

from evaluation import generation_metrics as gm
from evaluation import retrieval_metrics as rm
from evaluation import run_eval
from src import config
from src.query.context import build_context
from src.query.generator import REFUSAL
from src.query.observability import Trace
from src.query.pipeline import PipelineResult
from src.query.table_query import TableResult
from tests.helpers import search_hit

R = rm.Ref

# =====================================================================  retrieval metrics (precision / recall)


def test_precision_and_recall_when_the_one_relevant_source_is_among_six_chunks():
    retrieved = [R("t", "a.pdf", 1)] + [R("t", f"noise{i}.txt") for i in range(5)]
    s = rm.score_retrieval(retrieved, [R("t", "a.pdf")])
    assert s.recall == 1.0 and s.precision == pytest.approx(1 / 6) and s.relevant_chunks == 1 and s.missed == []


def test_several_chunks_of_the_relevant_file_raise_precision_not_recall():
    retrieved = [R("t", "a.pdf", 1), R("t", "a.pdf", 2), R("t", "a.pdf", 3), R("t", "x.txt")]
    s = rm.score_retrieval(retrieved, [R("t", "a.pdf")])
    assert s.precision == 0.75 and s.recall == 1.0


def test_recall_counts_how_many_of_the_relevant_sources_were_found_and_lists_the_missed_ones():
    s = rm.score_retrieval([R("t", "a.csv"), R("t", "z.txt")], [R("t", "a.csv"), R("u", "b.csv")])
    assert s.recall == 0.5 and s.precision == 0.5 and s.missed == [R("u", "b.csv")]


def test_a_page_label_requires_that_page_and_a_topic_must_match_too():
    assert rm.score_retrieval([R("t", "a.pdf", 1)], [R("t", "a.pdf", 2)]).recall == 0.0
    assert rm.score_retrieval([R("t", "a.pdf", 2)], [R("t", "a.pdf", 2)]).recall == 1.0
    assert rm.score_retrieval([R("t", "a.pdf", 7)], [R("t", "a.pdf")]).recall == 1.0         # no page in the label: any page
    assert rm.score_retrieval([R("other", "a.pdf")], [R("t", "a.pdf")]).recall == 0.0         # same name, different topic


def test_questions_without_relevant_sources_score_none_and_nothing_retrieved_scores_zero():
    none = rm.score_retrieval([R("t", "a")], [])
    assert none.precision is None and none.recall is None
    zero = rm.score_retrieval([], [R("t", "a")])
    assert zero.precision == 0.0 and zero.recall == 0.0 and zero.retrieved == 0


def test_k_limits_how_many_chunks_are_considered():
    retrieved = [R("t", "x"), R("t", "y"), R("t", "a")]
    assert rm.score_retrieval(retrieved, [R("t", "a")], k=2).recall == 0.0
    assert rm.score_retrieval(retrieved, [R("t", "a")], k=3).recall == 1.0


def test_summaries_ignore_questions_that_have_nothing_to_measure():
    scores = [rm.score_retrieval([R("t", "a")], [R("t", "a")]), rm.score_retrieval([R("t", "x")], [R("t", "a")]),
              rm.score_retrieval([R("t", "a")], [])]
    s = rm.summarize(scores)
    assert s == {"questions": 2, "precision": 0.5, "recall": 0.5, "full_recall": 1, "no_recall": 1}
    assert rm.mean([None, None]) is None and rm.mean([1.0, None, 0.0]) == 0.5
    assert rm.Ref.from_dict({"topic": "t", "source": "a", "page": 2}) == R("t", "a", 2) and str(R("t", "a", 2)) == "t/a p.2"


# =====================================================================  generation metrics


class FakeJudge:
    """Returns prepared verdicts by schema and remembers every prompt it was shown."""

    def __init__(self, claims=("supported", "supported", "unsupported"), facts=("present", "missing"), rating="full", fail=None):
        self.claims, self.facts, self.rating, self.fail, self.calls = claims, facts, rating, fail, []

    def __call__(self, prompt, schema, system=None, images=None):
        self.calls.append((schema.__name__, prompt, images))
        if self.fail == schema.__name__:
            raise RuntimeError("judge down")
        if schema is gm.FaithfulnessVerdict:
            return gm.FaithfulnessVerdict(claims=[gm.Claim(claim=f"c{i}", verdict=v) for i, v in enumerate(self.claims)])
        if schema is gm.CorrectnessVerdict:
            return gm.CorrectnessVerdict(facts=[gm.Fact(fact=f"f{i}", status=s) for i, s in enumerate(self.facts)])
        return gm.RelevanceVerdict(rating=self.rating, reason="because")


def test_faithfulness_is_the_share_of_supported_claims_and_citation_markers_are_not_judged():
    j = FakeJudge()
    s = gm.faithfulness("The total is 18 [1] and the shop is TEO HENG [C2].", "chunk text", j, images=[("cap", b"x", "image/jpeg")])
    assert s.score == pytest.approx(2 / 3) and [c["verdict"] for c in s.claims] == ["supported", "supported", "unsupported"]
    name, prompt, images = j.calls[0]
    assert "<evidence>\nchunk text\n</evidence>" in prompt and "[1]" not in prompt and "[C2]" not in prompt
    assert "The total is 18 and the shop is TEO HENG." in prompt and images == [("cap", b"x", "image/jpeg")]


def test_a_contradicted_claim_counts_against_faithfulness():
    assert gm.faithfulness("a", "e", FakeJudge(claims=("supported", "contradicted"))).score == 0.5


def test_correctness_is_the_share_of_reference_facts_stated_correctly():
    j = FakeJudge(facts=("present", "wrong", "missing", "present"))
    s = gm.correctness("q", "answer [1]", "the reference", j)
    assert s.score == 0.5 and "<reference>\nthe reference\n</reference>" in j.calls[0][1] and "[1]" not in j.calls[0][1]


def test_relevance_maps_the_rating_to_a_score():
    for rating, expected in (("full", 1.0), ("partial", 0.5), ("none", 0.0)):
        s = gm.answer_relevance("q", "a", FakeJudge(rating=rating))
        assert s.score == expected and s.rating == rating and s.reason == "because"


@pytest.mark.parametrize("which", ["FaithfulnessVerdict", "CorrectnessVerdict", "RelevanceVerdict"])
def test_a_failing_judge_gives_none_with_the_error_and_never_raises(which):
    j = FakeJudge(fail=which)
    s = {"FaithfulnessVerdict": lambda: gm.faithfulness("a", "e", j), "CorrectnessVerdict": lambda: gm.correctness("q", "a", "r", j),
         "RelevanceVerdict": lambda: gm.answer_relevance("q", "a", j)}[which]()
    assert s.score is None and "judge down" in s.error


def test_nothing_to_judge_gives_none_not_a_perfect_score():
    assert gm.faithfulness("a", "e", FakeJudge(claims=())).score is None
    assert gm.correctness("q", "a", "r", FakeJudge(facts=())).score is None


@pytest.mark.parametrize("answer,expected,ok", [
    ("There are 2,451 restaurants.", [{"value": 2451, "tol": 0}], True),
    ("The average is 2.438845.", [{"value": 2.4388, "tol": 0.01}], True),
    ("The average is 2.45.", [{"value": 2.4388, "tol": 0.01}], False),
    ("Total 259 sightings [1] across 5 cities.", [{"value": 259, "tol": 0}], True),
    ("It was 12 [5].", [{"value": 5, "tol": 0}], False),                 # the 5 in a [5] marker is not an answer
    ("Issued 23/01/2018.", [{"value": 2018, "tol": 0}], True),
    ("2451 and 2633", [{"value": 2451, "tol": 0}, {"value": 2633, "tol": 0}], True),
    ("only 2451", [{"value": 2451, "tol": 0}, {"value": 2633, "tol": 0}], False),
])
def test_numbers_check_tolerates_formatting_but_not_wrong_values(answer, expected, ok):
    assert gm.numbers_check(answer, expected).ok is ok


def test_numbers_check_reports_what_is_missing_and_does_nothing_without_expectations():
    assert gm.numbers_check("only 2451", [{"value": 2451}, {"value": 2633}]).missing == [2633]
    assert gm.numbers_check("anything", []).ok is None


def test_phrases_are_matched_ignoring_case_spacing_commas_and_markers():
    assert gm.contains_all("TEO  HENG stationery & books [1]", ["teo heng", "Stationery & Books"]).ok is True
    assert gm.contains_all("1,500 dollars", ["1500"]).ok is True
    r = gm.contains_all("Eldoret", ["Eldoret", "Nairobi"])
    assert r.ok is False and r.missing == ["Nairobi"] and gm.contains_all("x", []).ok is None


def test_score_generation_combines_everything_and_can_skip_the_judges():
    full = gm.score_generation("q", "2451 [1]", "ref", "ev", FakeJudge(), None, [{"value": 2451}], ["2451"])
    d = full.to_dict()
    assert d["faithfulness"] == pytest.approx(2 / 3) and d["correctness"] == 0.5 and d["relevance"] == 1.0
    assert d["numbers_ok"] is True and d["contains_ok"] is True and len(d["faithfulness_claims"]) == 3
    cheap = gm.score_generation("q", "2451", "ref", "ev", FakeJudge(), None, [{"value": 2451}], [], use_judge=False).to_dict()
    assert cheap["faithfulness"] is None and cheap["correctness"] is None and cheap["numbers_ok"] is True


# =====================================================================  the runner


def result(text="The answer is 18 [1].", answerable=True, hits=None, topics=("receipts",), calcs=(), stage=None, error=None):
    hits = hits if hits is not None else [search_hit("receipt text", topic="receipts", source="X1.jpg", n=1),
                                          search_hit("noise", topic="receipts", source="Y.jpg", n=2)]
    ctx = build_context(hits) if hits else None
    trace = Trace("q")
    trace.finish([])
    return PipelineResult("q", "q", text, answerable, [], stage, error, ctx, retrieved=list(hits), calculations=list(calcs),
                          topics=list(topics), trace=trace)


class FakePipeline:
    def __init__(self, outcomes):
        self.outcomes, self.asked = outcomes, []

    def answer(self, question, history=None):
        self.asked.append((question, history))
        out = self.outcomes[question] if isinstance(self.outcomes, dict) else self.outcomes
        if isinstance(out, Exception):
            raise out
        return out


CASE = {"id": "q1", "category": "image_ocr", "question": "total?", "reference": "18", "expected_numbers": [{"value": 18, "tol": 0}],
        "must_contain": ["answer"], "expected_topics": ["receipts"], "relevant": [{"topic": "receipts", "source": "X1.jpg"}]}


def test_an_answered_question_is_scored_on_retrieval_routing_and_generation():
    judge = FakeJudge()
    rec = run_eval.evaluate_question(CASE, FakePipeline(result()), judge)
    assert rec["retrieval"]["recall"] == 1.0 and rec["retrieval"]["precision"] == 0.5
    assert rec["retrieval_in_context"]["recall"] == 1.0 and rec["routing_ok"] is True and rec["false_refusal"] is False
    g = rec["generation"]
    assert g["numbers_ok"] is True and g["contains_ok"] is True and g["relevance"] == 1.0 and g["correctness"] == 0.5
    assert rec["answerable"] is True and rec["id"] == "q1" and rec["seconds"] >= 0


def test_the_judges_token_cost_is_recorded_separately_from_the_pipelines():
    from src.core.llm import Usage, record_usage

    class CostlyJudge(FakeJudge):
        def __call__(self, prompt, schema, system=None, images=None):
            record_usage(Usage(100, 10, 5))                                  # as the real judge does on every call
            return super().__call__(prompt, schema, system, images)

    rec = run_eval.evaluate_question(CASE, FakePipeline(result()), CostlyJudge())
    assert rec["judge_calls"] == 3 and rec["judge_tokens"] == 3 * 115
    cheap = run_eval.evaluate_question(CASE, FakePipeline(result()), CostlyJudge(), use_judge=False)
    assert cheap["judge_calls"] == 0 and cheap["judge_tokens"] == 0
    cost = run_eval.summarize([rec, cheap])["cost"]
    assert cost["judge_tokens"] == 345 and cost["model_calls"] == 3


def test_the_faithfulness_judge_is_given_the_context_and_the_calculation_results():
    judge = FakeJudge()
    calc = TableResult("C1", "T1", "t", "x.csv", "main", "count of rows", True, value=7, rows_total=10, rows_matched=7)
    run_eval.evaluate_question(CASE, FakePipeline(result(calcs=[calc])), judge)
    prompt = next(p for name, p, _ in judge.calls if name == "FaithfulnessVerdict")
    assert "receipt text" in prompt and "Calculation results:" in prompt and "result: 7" in prompt


def test_no_judge_mode_makes_no_judge_calls_but_keeps_the_cheap_checks():
    judge = FakeJudge()
    rec = run_eval.evaluate_question(CASE, FakePipeline(result()), judge, use_judge=False)
    assert judge.calls == [] and rec["generation"]["faithfulness"] is None and rec["generation"]["numbers_ok"] is True


def test_routing_is_judged_by_the_set_of_topics():
    wrong = run_eval.evaluate_question(CASE, FakePipeline(result(topics=("restaurants",))), FakeJudge(), use_judge=False)
    assert wrong["routing_ok"] is False
    no_expectation = {k: v for k, v in CASE.items() if k != "expected_topics"}
    assert "routing_ok" not in run_eval.evaluate_question(no_expectation, FakePipeline(result()), FakeJudge(), use_judge=False)


REFUSE = {"id": "r1", "category": "refusal", "question": "pizza?", "should_refuse": True, "reference": "decline", "relevant": []}


def test_refusal_questions_must_be_declined_and_are_not_judged():
    declined = run_eval.evaluate_question(REFUSE, FakePipeline(result(REFUSAL, False, hits=[], topics=(), stage="router")), FakeJudge())
    assert declined["refusal_ok"] is True and "generation" not in declined and declined["retrieval"]["recall"] is None
    answered = run_eval.evaluate_question(REFUSE, FakePipeline(result("Pizza dough is...")), FakeJudge())
    assert answered["refusal_ok"] is False


def test_declining_an_answerable_question_is_a_false_refusal_with_zero_scores():
    rec = run_eval.evaluate_question(CASE, FakePipeline(result(REFUSAL, False, stage="generator")), FakeJudge())
    assert rec["false_refusal"] is True and rec["generation"]["correctness"] == 0.0 and rec["generation"]["numbers_ok"] is False


def test_a_crashing_pipeline_becomes_an_error_record():
    rec = run_eval.evaluate_question(CASE, FakePipeline(RuntimeError("boom")), FakeJudge())
    assert rec["error"] == "RuntimeError: boom" and "answer" not in rec and rec["id"] == "q1"


def test_history_is_passed_on_for_follow_up_questions():
    pipe = FakePipeline(result())
    run_eval.evaluate_question({**CASE, "history": [{"role": "user", "content": "hi"}]}, pipe, FakeJudge(), use_judge=False)
    assert pipe.asked[0][1] == [{"role": "user", "content": "hi"}]


def test_summary_averages_by_category_and_lists_failures(tmp_path):
    good = run_eval.evaluate_question(CASE, FakePipeline(result()), FakeJudge(claims=("supported",), facts=("present",)))
    bad = run_eval.evaluate_question({**CASE, "id": "q2", "category": "table_calc"},
                                     FakePipeline(result("It is 99.", hits=[search_hit("x", topic="receipts", source="Z.jpg", n=1)])),
                                     FakeJudge(claims=("unsupported",), facts=("missing",)))
    ref_ok = run_eval.evaluate_question(REFUSE, FakePipeline(result(REFUSAL, False, hits=[], topics=(), stage="router")), FakeJudge())
    ref_bad = run_eval.evaluate_question({**REFUSE, "id": "r2"}, FakePipeline(result("Pizza is...")), FakeJudge())
    crashed = run_eval.evaluate_question({**CASE, "id": "q3"}, FakePipeline(RuntimeError("x")), FakeJudge())
    report = run_eval.build_report([good, bad, ref_ok, ref_bad, crashed], {"model": "m"})

    s = report["summary"]
    assert (s["questions"], s["answerable"], s["should_refuse"], s["crashed"]) == (5, 2, 2, 1)
    assert s["retrieval"]["recall"] == 0.5 and s["generation"]["correctness"] == 0.5 and s["generation"]["faithfulness"] == 0.5
    assert s["refusal_accuracy"] == 0.5 and s["false_refusal_rate"] == 0.0
    assert set(report["by_category"]) == {"image_ocr", "table_calc", "refusal"}
    why = {f["id"]: " | ".join(f["why"]) for f in report["failures"]}
    assert set(why) == {"q2", "r2", "q3"}
    assert "expected number(s) missing" in why["q2"] and "retrieval missed every relevant source" in why["q2"]
    assert "should have declined" in why["r2"] and "crashed" in why["q3"]


def test_questions_run_in_parallel_but_results_keep_the_question_order():
    cases = [{**CASE, "id": f"q{i}", "question": f"question {i}"} for i in range(8)]
    pipe = FakePipeline({c["question"]: result() for c in cases})
    records = run_eval.run(cases, pipe, FakeJudge(), use_judge=False, workers=3)
    assert [r["id"] for r in records] == [f"q{i}" for i in range(8)]
    assert [r["id"] for r in run_eval.run(cases, pipe, FakeJudge(), use_judge=False, workers=1)] == [f"q{i}" for i in range(8)]


def test_main_writes_the_results_file_and_honours_the_filters(tmp_path, monkeypatch, capsys):
    import src.query.pipeline
    pipe = FakePipeline(result())
    monkeypatch.setattr(src.query.pipeline, "RagPipeline", lambda: pipe)
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"questions": [CASE, {**CASE, "id": "q2", "category": "followup", "question": "later?"}]}))
    out = tmp_path / "out" / "results.json"

    assert run_eval.main(["--testset", str(testset), "--out", str(out), "--no-judge", "--categories", "followup", "--workers", "1"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [q["id"] for q in report["questions"]] == ["q2"] and report["run"]["judge"] is False
    assert report["run"]["model"] == config.LLM_MODEL and "summary" in report and "by_category" in report
    assert "1 questions" in capsys.readouterr().out
    assert run_eval.main(["--testset", str(testset), "--out", str(out), "--ids", "nope", "--no-judge"]) == 2


# =====================================================================  the question set itself
QUESTIONS = run_eval.load_testset()
CATEGORIES = {"text_docx", "text_html", "text_lyrics", "text_markdown", "pdf_text", "pdf_table", "pdf_chart", "scanned_pdf",
              "image_ocr", "image_photo", "table_lookup", "table_stats", "table_calc", "table_group", "multi_topic",
              "cross_modal", "followup", "refusal"}
DATA = config.DATA_DIR
MAX_QUESTIONS = 30                  # a deliberate ceiling: a full judged run is roughly 7 model calls per question


def test_every_question_is_well_formed():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert len(CATEGORIES) <= len(QUESTIONS) <= MAX_QUESTIONS, "keep the set small: every run and every judge costs API tokens"
    assert len({q["question"] for q in QUESTIONS if not q.get("history")}) == len([q for q in QUESTIONS if not q.get("history")])
    for q in QUESTIONS:
        assert {"id", "category", "question", "reference"} <= set(q), q["id"]
        assert q["category"] in CATEGORIES, q["id"]
        if q.get("should_refuse"):
            assert q["relevant"] == [] and q["category"] == "refusal", q["id"]
        else:
            assert q["relevant"] and q.get("expected_topics") and (q.get("expected_numbers") or q.get("must_contain")), q["id"]
        for n in q.get("expected_numbers", []):
            assert isinstance(n["value"], (int, float)) and n.get("tol", 0) >= 0, q["id"]
        for t in q.get("history", []):
            assert t["role"] in ("user", "assistant") and t["content"], q["id"]
        for r in q["relevant"]:
            assert r["topic"] and r["source"], q["id"]
    assert {q["category"] for q in QUESTIONS} == CATEGORIES                                 # every capability is covered
    assert all(q["history"] for q in QUESTIONS if q["category"] == "followup")


@pytest.mark.skipif(not DATA.exists(), reason="sample data not present")
def test_every_relevant_source_and_expected_topic_really_exists_in_the_data():
    topics = {p.name for p in DATA.iterdir() if p.is_dir()}
    for q in QUESTIONS:
        for t in q.get("expected_topics", []):
            assert t in topics, (q["id"], t)
        for r in q["relevant"]:
            assert (DATA / r["topic"] / r["source"]).is_file(), (q["id"], r)


@pytest.mark.skipif(not DATA.exists(), reason="sample data not present")
def test_the_reference_numbers_match_the_data_recomputed_with_plain_pandas():
    num = lambda s: float(re.sub(r"[^0-9.]", "", str(s)))
    z = pd.read_csv(DATA / "restaurants" / "zomato_restaurants.csv")
    t = pd.read_csv(DATA / "airline_tweets" / "airline_tweets.csv")
    rc = pd.read_csv(DATA / "receipts" / "receipts_ground_truth.csv", dtype=str)
    neg = t[t.airline_sentiment == "negative"]
    truth = {
        "rest_delivery": int((z["Has Online delivery"] == "Yes").sum()),
        "rest_sao_paulo": int((z.City == "Sí£o Paulo").sum()),
        "air_top": int(neg.groupby("airline").size().max()),
        "receipts_sum": float(rc.total.map(num).sum()),
    }
    by_id = {q["id"]: q for q in QUESTIONS}
    for qid, actual in truth.items():
        expected = by_id[qid]["expected_numbers"][0]
        assert abs(actual - expected["value"]) <= max(expected.get("tol", 0), 1e-9), (qid, actual, expected)
    multi = by_id["multi_delivery_united"]["expected_numbers"]
    assert [m["value"] for m in multi] == [truth["rest_delivery"], truth["air_top"]]
    assert neg.groupby("airline").size().idxmax() == "United" and neg.groupby("airline").size().idxmin() == "Virgin America"
