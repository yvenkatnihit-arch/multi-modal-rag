from types import SimpleNamespace

import ask
from src.query.citations import Source
from src.query.generator import REFUSAL
from src.query.observability import Trace
from src.query.pipeline import PipelineResult


class FakePipeline:
    def __init__(self):
        self.calls = []

    def answer(self, question, history=None):
        self.calls.append((question, list(history or [])))                  # a snapshot: the chat mutates its list later
        src = Source([2], "receipts", "r.jpg", "image", "", "receipts · r.jpg · image")
        trace = Trace(question)
        trace.finish([])
        if "pizza" in question:
            return PipelineResult(question, question, REFUSAL, False, refusal_stage="router", trace=trace)
        return PipelineResult(question, question, f"answer to {question} [2]", True, [src], trace=trace)


def run_chat(inputs, **kw):
    lines, out = iter(inputs), []
    pipe = FakePipeline()

    def read(prompt):                                   # like input(): raises EOFError when the input is used up
        line = next(lines, None)
        if line is None:
            raise EOFError
        return line

    history = ask.chat(pipe, read=read, write=out.append, **kw)
    return pipe, history, "\n".join(out)


def test_conversation_history_is_passed_to_each_question():
    pipe, history, _ = run_chat(["first", "second", "/quit"])
    assert pipe.calls[0] == ("first", [])
    assert pipe.calls[1][0] == "second" and pipe.calls[1][1] == [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "answer to first [2]"}]
    assert len(history) == 4


def test_clear_forgets_the_conversation():
    pipe, history, out = run_chat(["first", "/clear", "third", "/quit"])
    assert pipe.calls[1] == ("third", []) and len(history) == 2 and "(conversation cleared)" in out


def test_sources_are_listed_with_their_chunk_numbers_and_refusals_have_none():
    _, _, out = run_chat(["receipt total?", "recipe for pizza?", "/quit"])
    assert "Sources:\n  [2] receipts · r.jpg · image" in out
    assert out.count("Sources:") == 1 and REFUSAL in out


def test_a_calculation_source_is_listed_by_its_C_number():
    calc = Source([], "restaurants", "zomato.csv", "table", "calculated over 5,473 of 9,551 rows",
                  "restaurants · zomato.csv · calculation: average of 'Aggregate rating' (5,473 of 9,551 rows) · table",
                  calc_refs=["C1"], calculation="average of 'Aggregate rating'")
    chunk = Source([2, 5], "receipts", "r.jpg", "image", "", "receipts · r.jpg · image")
    result = PipelineResult("q", "q", "Answer [C1][2]", True, [chunk, calc])
    out = ask.format_result(result)
    assert "  [2][5] receipts · r.jpg · image" in out and "  [C1] restaurants · zomato.csv · calculation:" in out


def test_trace_command_toggles_the_trace_output():
    _, _, out = run_chat(["/trace", "q one", "/trace", "q two", "/quit"])
    assert "(trace on)" in out and "(trace off)" in out
    assert out.count("trace:") == 1                                         # shown for "q one" only


def test_blank_lines_unknown_commands_and_end_of_input_are_handled():
    pipe, _, out = run_chat(["", "   ", "/nonsense", "real question"])   # input ends without /quit: EOFError -> exit
    assert [c[0] for c in pipe.calls] == ["real question"] and "Commands:" in out


def test_topics_command_lists_the_topics():
    topics = [SimpleNamespace(id="receipts", description="Scanned shop receipts " * 10)]
    _, _, out = run_chat(["/topics", "/quit"], topics=topics)
    assert "receipts" in out and "Scanned shop receipts" in out
