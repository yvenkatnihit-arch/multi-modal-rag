import pandas as pd
import pytest

from src import config
from src.core import assets
from src.query.citations import build_citations
from src.query.context import build_context
from src.query.generator import MAX_CALC_ROUNDS, REFUSAL, GeneratedAnswer, generate_answer
from src.core.llm import Usage
from src.query.table_query import Filter, TableInfo, TableRequest, TableResult, build_catalog
from tests.helpers import search_hit

CTX = build_context([search_hit("table summary: 6 rows x 3 columns", topic="t", source="sales.csv", modality="table",
                                table_id="main", part="summary", n=1)])
TABLES = [TableInfo("T1", "t", "sales.csv", "main", 6, {"city": "text", "rating": "decimal"})]
REQ = TableRequest(table="T1", operation="mean", column="rating", filters=[Filter(column="city", op="==", value="Mumbai")])


def calc(ref="C1", ok=True, value=4.75):
    return TableResult(ref, "T1", "t", "sales.csv", "main", "average (mean) of 'rating' where city == 'Mumbai'", ok,
                       value=value if ok else None, rows_total=6, rows_matched=3, error=None if ok else "bad column")


class Script:
    """A fake LLM that returns a prepared reply per call and records every prompt."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt, schema, system=None, thinking_budget=0):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1], Usage(100, 10, 5)


def ask(script, tables=TABLES, results=None):
    ran = []

    def run(reqs, start):
        ran.append((len(reqs), start))
        return [calc(f"C{start + i}") for i in range(len(reqs))] if results is None else results

    return generate_answer("average rating in Mumbai?", CTX, script, tables=tables, run_requests=run), ran


def want():
    return GeneratedAnswer(answerable=True, answer="", table_requests=[REQ])


def final(text="The average is 4.75 [C1].", calcs=(1,)):
    return GeneratedAnswer(answerable=True, answer=text, used_calculations=list(calcs))


# ------------------------------------------------------------------ the loop
def test_without_tables_one_call_is_made_and_requests_are_ignored():
    s = Script(GeneratedAnswer(answerable=True, answer="Answer [1].", used_chunks=[1], table_requests=[REQ]))
    a, ran = ask(s, tables=[])
    assert len(s.prompts) == 1 and ran == [] and "<tables>" not in s.prompts[0] and a.text == "Answer [1]."


def test_with_tables_but_no_request_it_is_still_one_call_and_the_catalog_is_in_the_prompt():
    s = Script(GeneratedAnswer(answerable=True, answer="Stated in the summary [1].", used_chunks=[1]))
    a, ran = ask(s)
    assert len(s.prompts) == 1 and ran == [] and a.calculations == []
    assert "<tables>\nT1: t/sales.csv, 6 rows. Columns: city (text), rating (decimal)\n</tables>" in s.prompts[0]
    assert "<calculations>" not in s.prompts[0] and "No further calculations" not in s.prompts[0]


def test_a_request_is_run_and_its_result_goes_back_to_the_model_for_the_final_answer():
    s = Script(want(), final())
    a, ran = ask(s)
    assert len(s.prompts) == 2 and ran == [(1, 1)]
    assert "<calculations>\nC1 [T1: t/sales.csv] average (mean) of 'rating' where city == 'Mumbai'" in s.prompts[1]
    assert "result: 4.75" in s.prompts[1] and "<calculations>" not in s.prompts[0]
    assert a.answerable and a.text == "The average is 4.75 [C1]." and a.used_calculations == [1]
    assert [c.ref for c in a.calculations] == ["C1"]
    assert a.usage == Usage(200, 20, 10)                                           # both calls are counted


def test_a_second_round_continues_the_calculation_numbering():
    s = Script(want(), want(), final("It is 4.75 [C2].", calcs=(2,)))
    a, ran = ask(s)
    assert ran == [(1, 1), (1, 2)] and [c.ref for c in a.calculations] == ["C1", "C2"] and len(s.prompts) == 3


def test_the_number_of_rounds_is_capped_and_the_last_prompt_says_so():
    s = Script(want())                                                              # the model never stops asking
    a, ran = ask(s)
    assert len(s.prompts) == MAX_CALC_ROUNDS + 1 == 3 and len(ran) == MAX_CALC_ROUNDS
    assert "No further calculations can be requested" in s.prompts[-1] and "No further" not in s.prompts[0]
    assert not a.answerable and a.refusal_reason == "empty_answer" and a.text == REFUSAL    # nothing usable came out
    assert len(a.calculations) == 2                                                 # kept, for the trace


def test_a_refusal_after_calculations_keeps_them_for_the_trace_but_cites_nothing():
    s = Script(want(), GeneratedAnswer(answerable=False, answer="Mumbai has no ratings."))
    a, _ = ask(s)
    assert not a.answerable and a.refusal_reason == "unsupported" and len(a.calculations) == 1
    assert build_citations(a, CTX).sources == []


# ------------------------------------------------------------------ citing calculations
def test_a_calculation_is_cited_with_what_was_computed_and_over_how_many_rows():
    s = Script(want(), final())
    a, _ = ask(s)
    c = build_citations(a, CTX)
    [src] = c.sources
    assert src.refs == ["C1"] and src.numbers == [] and src.modality == "table" and not c.uncited
    assert src.display == ("t · sales.csv · calculation: average (mean) of 'rating' where city == 'Mumbai' (3 of 6 rows) · table")
    assert src.calculation.startswith("average (mean)") and src.location == "calculated over 3 of 6 rows"


def test_chunks_and_calculations_are_cited_together():
    s = Script(want(), GeneratedAnswer(answerable=True, answer="Mumbai averages 4.75 [C1] across 6 rows [1].",
                                      used_chunks=[1], used_calculations=[1]))
    a, _ = ask(s)
    assert [s.refs for s in build_citations(a, CTX).sources] == [["1"], ["C1"]]


def test_failed_or_invented_calculations_are_never_cited():
    s = Script(want(), final("It is 4.75 [C1] or maybe [C7].", calcs=(1, 7)))
    a, _ = ask(s, results=[calc("C1", ok=False)])
    c = build_citations(a, CTX)
    assert c.sources == [] and c.invalid_calculations == [1, 7] and c.uncited


# ------------------------------------------------------------------ the real calculator, with only the LLM faked
def test_end_to_end_the_requested_figure_is_computed_from_the_stored_table(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")
    df = pd.DataFrame({"city": ["New Delhi", "New Delhi", "Mumbai", "Mumbai"], "rating": [4.0, 3.0, 5.0, 4.5]})
    assets.save_table(df, "t", "sales.csv", "main")
    catalog = build_catalog(["t"])
    s = Script(GeneratedAnswer(answerable=True, answer="", table_requests=[TableRequest(
                   table=catalog[0].ref, operation="mean", column="rating", filters=[Filter(column="CITY", op="==", value="new delhi")])]),
               final("The average rating in New Delhi is 3.5 [C1]."))
    a = generate_answer("average rating in New Delhi?", CTX, s, tables=catalog)         # default run_requests = the real engine
    assert a.calculations[0].value == 3.5 and a.calculations[0].rows_matched == 2
    assert "result: 3.5" in s.prompts[1] and "computed over 2 of 4 rows" in s.prompts[1]
    assert a.text == "The average rating in New Delhi is 3.5 [C1]."


def test_markers_that_point_at_nothing_are_removed_from_the_answer_text():
    from src.query.generator import Answer
    ans = Answer("Total is 259 [C1], from the table [1][9], and a note [C7].", True, [1], calculations=[calc("C2", ok=False)])
    c = build_citations(ans, CTX)
    assert c.text == "Total is 259, from the table [1], and a note."                 # [C1] (never run), [9], [C7] all gone
    refusal = Answer(f"{REFUSAL} Nothing [3].", False)
    assert build_citations(refusal, CTX).text == ""                                  # refusals are left untouched by the pipeline
    good = Answer("It is 4.75 [C1].", True, [], calculations=[calc("C1")])
    assert build_citations(good, CTX).text == "It is 4.75 [C1]."                     # real markers stay


def test_the_system_prompt_explains_when_and_how_to_request_calculations():
    from src.query.generator import SYSTEM
    for phrase in ("table_requests", "<calculations>", "closest existing values", "do NOT estimate", "[C1]"):
        assert phrase in SYSTEM
