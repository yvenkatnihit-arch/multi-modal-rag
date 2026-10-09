"""Run the question set through the pipeline and score it.

    python -m evaluation.run_eval                        everything (answers + LLM judges: a few hundred Gemini calls)
    python -m evaluation.run_eval --no-judge             retrieval, routing, refusals and deterministic checks only
    python -m evaluation.run_eval --categories refusal followup
    python -m evaluation.run_eval --ids rest_delivery air_top --workers 1

Writes evaluation/results/results.json: the summary (overall and per category) and one record per question with its
answer, sources, retrieval scores, judge verdicts (claim by claim) and timings.

What is scored, per question:
    retrieval   precision / recall of the retrieved chunks vs the labelled relevant sources   (retrieval_metrics.py)
    routing     did the router choose exactly the expected topics
    answerable  faithfulness / correctness / answer relevance, plus the number and phrase checks (generation_metrics.py)
    refusals    questions flagged should_refuse must be declined; any other question that is declined is a FALSE refusal
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # allows `python evaluation/run_eval.py` as well as `-m`
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402  (first: installs the gRPC telemetry stub before chromadb is imported)
from src.core.llm import track_usage  # noqa: E402

from evaluation import generation_metrics as gm  # noqa: E402
from evaluation import retrieval_metrics as rm  # noqa: E402

TESTSET = Path(__file__).resolve().parent / "testset.json"
RESULTS = Path(__file__).resolve().parent / "results" / "results.json"

LOW_CORRECTNESS = 0.67        # below this an answered question is listed as a failure
LOW_FAITHFULNESS = 0.8


def load_testset(path: Path = TESTSET) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def _refs(hits_or_items) -> list[rm.Ref]:
    out = []
    for h in hits_or_items:
        m = h.metadata
        out.append(rm.Ref(m.get("topic_id", getattr(h, "topic_id", "")), m.get("source", ""), m.get("page"), None, h.text))
    return out


def evaluate_retrieval(case: dict, searcher, embed=None) -> dict | None:
    """Retrieval ONLY, with no LLM calls at all: embed the question, search the expected topics, score the chunks.
    Costs a handful of embedding calls, so it is the cheap way to compare search changes before and after.
    Returns None for questions with nothing to measure (refusals). A follow-up may carry a `standalone` wording."""
    from src.query.context import build_context
    from src.core.embeddings import embed_query
    from src.query.merger import merge

    topics, relevant = case.get("expected_topics"), [rm.Ref.from_dict(r) for r in case.get("relevant", [])]
    if not topics or not relevant:
        return None
    question = case.get("standalone", case["question"])
    qvec = (embed or embed_query)(question)
    merged = merge({t: searcher.search(t, question, query_vector=qvec) for t in topics})
    return {"id": case["id"], "category": case["category"], "question": question, "should_refuse": False,
            "retrieval": rm.score_retrieval(_refs(merged.hits), relevant).to_dict(),
            "retrieval_in_context": rm.score_retrieval(_refs(build_context(merged.hits).items), relevant).to_dict()}


def _calculation_text(result) -> str:
    return "\n".join(c.to_text() for c in result.calculations if c.ok)


def evaluate_question(case: dict, pipeline, judge=gm.default_judge, use_judge: bool = True) -> dict:
    """Run one question and score it. Never raises: a crash becomes an error record."""
    t0 = time.perf_counter()
    rec: dict = {"id": case["id"], "category": case["category"], "question": case["question"],
                 "reference": case.get("reference"), "should_refuse": bool(case.get("should_refuse"))}
    try:
        result = pipeline.answer(case["question"], case.get("history"))
    except Exception as e:
        return {**rec, "error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t0, 2)}

    relevant = [rm.Ref.from_dict(r) for r in case.get("relevant", [])]
    retrieved = _refs(result.retrieved)
    in_context = _refs(result.context.items) if result.context else []
    refused = not result.answerable

    rec.update({
        "answer": result.text, "answerable": result.answerable, "refusal_stage": result.refusal_stage,
        "error": result.error, "sources": [s.display for s in result.sources], "routed": result.topics,
        "retrieval": rm.score_retrieval(retrieved, relevant).to_dict(),
        "retrieval_in_context": rm.score_retrieval(in_context, relevant).to_dict(),
        "calculations": [c.to_dict() for c in result.calculations],
        "seconds": round(result.trace.total_seconds, 2) if result.trace else round(time.perf_counter() - t0, 2),
        "llm_calls": result.trace.llm_calls if result.trace else None,
        "tokens": result.trace.tokens if result.trace else None,
    })
    if "expected_topics" in case:
        rec["routing_ok"] = set(result.topics) == set(case["expected_topics"])

    if case.get("should_refuse"):
        rec["refusal_ok"] = refused                                   # declined, as it should be
        return rec
    rec["false_refusal"] = refused
    if refused:
        rec["generation"] = {"faithfulness": None, "correctness": 0.0, "relevance": 0.0,
                             "numbers_ok": False if case.get("expected_numbers") else None,
                             "contains_ok": False if case.get("must_contain") else None}
        return rec

    images = None
    if use_judge and result.context:                                   # the judge sees the same pictures the model saw
        from src.query.pictures import select_pictures
        images = [p.as_llm_input() for p in select_pictures(result.context)] or None
    evidence = (result.context.text if result.context else "") + (
        "\n\nCalculation results:\n" + _calculation_text(result) if result.calculations else "")
    with track_usage() as judge_calls:                                 # the judges' own cost, kept apart from the pipeline's
        scores = gm.score_generation(
            case["question"], result.text, case.get("reference", ""), evidence, judge, images,
            case.get("expected_numbers", ()), case.get("must_contain", ()), use_judge,
        )
    rec["generation"] = scores.to_dict()
    rec["judge_calls"] = len(judge_calls)
    rec["judge_tokens"] = sum(u.prompt_tokens + u.output_tokens + u.thinking_tokens for u in judge_calls)
    return rec


# ------------------------------------------------------------------ summaries
def _avg(values):
    return rm.mean(values)


def _rate(flags):
    flags = [f for f in flags if f is not None]
    return sum(flags) / len(flags) if flags else None


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if not r.get("error") or r.get("answer") is not None]
    crashed = [r for r in records if r.get("error") and r.get("answer") is None]
    answerable = [r for r in ok if not r["should_refuse"]]
    refusals = [r for r in ok if r["should_refuse"]]
    gen = [r["generation"] for r in answerable if r.get("generation")]
    secs = [r["seconds"] for r in ok if r.get("seconds") is not None]
    scores = [rm.RetrievalScore(r["retrieval"]["precision"], r["retrieval"]["recall"], r["retrieval"]["retrieved"],
                                r["retrieval"]["relevant_chunks"], r["retrieval"]["relevant_total"],
                                r["retrieval"]["relevant_found"]) for r in ok if "retrieval" in r]
    ctx_scores = [rm.RetrievalScore(r["retrieval_in_context"]["precision"], r["retrieval_in_context"]["recall"], 0, 0, 0, 0)
                  for r in ok if "retrieval_in_context" in r]
    return {
        "questions": len(records), "crashed": len(crashed), "answerable": len(answerable), "should_refuse": len(refusals),
        "retrieval": rm.summarize(scores),
        "retrieval_in_context": {"recall": _avg(s.recall for s in ctx_scores if s.recall is not None),
                                 "precision": _avg(s.precision for s in ctx_scores if s.precision is not None)},
        "routing_accuracy": _rate(r.get("routing_ok") for r in ok),
        "generation": {
            "faithfulness": _avg(g.get("faithfulness") for g in gen),
            "correctness": _avg(g.get("correctness") for g in gen),
            "answer_relevance": _avg(g.get("relevance") for g in gen),
            "numbers_check_pass_rate": _rate(g.get("numbers_ok") for g in gen),
            "phrase_check_pass_rate": _rate(g.get("contains_ok") for g in gen),
        },
        "refusal_accuracy": _rate(r.get("refusal_ok") for r in refusals),
        "false_refusal_rate": _rate(r.get("false_refusal") for r in answerable),
        "seconds": {"mean": _avg(secs), "p95": sorted(secs)[int(0.95 * (len(secs) - 1))] if secs else None},
        "tokens_per_question": _avg((r["tokens"]["prompt"] + r["tokens"]["output"] + r["tokens"]["thinking"])
                                    for r in ok if r.get("tokens")),
        "cost": {                                                   # what the run spent (model tokens and calls)
            "pipeline_tokens": sum(r["tokens"]["prompt"] + r["tokens"]["output"] + r["tokens"]["thinking"] for r in ok if r.get("tokens")),
            "judge_tokens": sum(r.get("judge_tokens", 0) for r in ok),
            "model_calls": sum((r.get("llm_calls") or 0) + r.get("judge_calls", 0) for r in ok),
        },
    }


def failures(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        why = []
        if r.get("error") and r.get("answer") is None:
            why.append(f"crashed: {r['error']}")
        elif r["should_refuse"] and not r.get("refusal_ok"):
            why.append("answered a question it should have declined")
        elif not r["should_refuse"]:
            g = r.get("generation") or {}
            if r.get("false_refusal"):
                why.append("declined a question it should have answered")
            if g.get("numbers_ok") is False:
                why.append(f"expected number(s) missing: {g.get('numbers_missing')}")
            if g.get("contains_ok") is False:
                why.append(f"expected phrase(s) missing: {g.get('contains_missing')}")
            if not r.get("false_refusal"):
                if g.get("correctness") is not None and g["correctness"] < LOW_CORRECTNESS:
                    why.append(f"low correctness {g['correctness']:.2f}")
                if g.get("faithfulness") is not None and g["faithfulness"] < LOW_FAITHFULNESS:
                    why.append(f"low faithfulness {g['faithfulness']:.2f}")
            if (r.get("retrieval") or {}).get("recall") == 0.0:
                why.append("retrieval missed every relevant source")
            if r.get("routing_ok") is False:
                why.append(f"routed to {r.get('routed')}")
        if why:
            out.append({"id": r["id"], "category": r["category"], "question": r["question"], "why": why})
    return out


def build_report(records: list[dict], meta: dict) -> dict:
    cats = sorted({r["category"] for r in records})
    return {"run": meta, "summary": summarize(records),
            "by_category": {c: summarize([r for r in records if r["category"] == c]) for c in cats},
            "failures": failures(records), "questions": records}


# ------------------------------------------------------------------ output
def _f(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x * 100:4.0f}%" if pct else f"{x:5.2f}"


def print_report(report: dict) -> None:
    s = report["summary"]
    g = s["generation"]
    print(f"\n{s['questions']} questions ({s['answerable']} answerable, {s['should_refuse']} to refuse, {s['crashed']} crashed)")
    print(f"  retrieval      recall {_f(s['retrieval']['recall'])}   precision {_f(s['retrieval']['precision'])}   "
          f"(full recall on {s['retrieval']['full_recall']}/{s['retrieval']['questions']}, none on {s['retrieval']['no_recall']})")
    print(f"  in the prompt  recall {_f(s['retrieval_in_context']['recall'])}   precision {_f(s['retrieval_in_context']['precision'])}")
    print(f"  routing        accuracy {_f(s['routing_accuracy'], True)}")
    print(f"  generation     faithfulness {_f(g['faithfulness'])}   correctness {_f(g['correctness'])}   relevance {_f(g['answer_relevance'])}")
    print(f"  checks         numbers {_f(g['numbers_check_pass_rate'], True)}   phrases {_f(g['phrase_check_pass_rate'], True)}")
    print(f"  refusals       correct {_f(s['refusal_accuracy'], True)}   false refusals {_f(s['false_refusal_rate'], True)}")
    print(f"  speed          mean {_f(s['seconds']['mean'])}s   p95 {_f(s['seconds']['p95'])}s")
    c = s["cost"]
    print(f"  cost           {c['model_calls']} model calls | tokens: pipeline {c['pipeline_tokens']:,} + judges {c['judge_tokens']:,} "
          f"= {c['pipeline_tokens'] + c['judge_tokens']:,}")
    print(f"\n  {'category':<16}{'n':>3}  {'recall':>6} {'prec':>6} {'faith':>6} {'corr':>6} {'relev':>6} {'refuse':>6}")
    for c, v in report["by_category"].items():
        print(f"  {c:<16}{v['questions']:>3}  {_f(v['retrieval']['recall']):>6} {_f(v['retrieval']['precision']):>6} "
              f"{_f(v['generation']['faithfulness']):>6} {_f(v['generation']['correctness']):>6} "
              f"{_f(v['generation']['answer_relevance']):>6} {_f(v['refusal_accuracy'], True):>6}")
    if report["failures"]:
        print(f"\n  {len(report['failures'])} question(s) need a look:")
        for f in report["failures"]:
            print(f"   - {f['id']:<24} {'; '.join(f['why'])}")
    else:
        print("\n  no failures")


def run(cases, pipeline, judge=gm.default_judge, use_judge=True, workers=3) -> list[dict]:
    if workers <= 1:
        return [evaluate_question(c, pipeline, judge, use_judge) for c in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:             # results keep the question order
        return list(pool.map(lambda c: evaluate_question(c, pipeline, judge, use_judge), cases))


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--categories", nargs="*", help="only these categories")
    ap.add_argument("--limit", type=int, help="only the first N after filtering")
    ap.add_argument("--workers", type=int, default=3, help="questions run in parallel (default 3; 1 = one at a time)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judges (cheap: retrieval, routing, refusals, checks)")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="no LLM calls at all: score only the search, using each question's expected topics (embeddings only)")
    ap.add_argument("--testset", type=Path, default=TESTSET)
    ap.add_argument("--out", type=Path, default=None, help="default: results/results.json (results/retrieval_only.json with --retrieval-only)")
    args = ap.parse_args(argv)
    args.out = args.out or (RESULTS.with_name("retrieval_only.json") if args.retrieval_only else RESULTS)

    cases = load_testset(args.testset)
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    if args.categories:
        cases = [c for c in cases if c["category"] in args.categories]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no questions selected")
        return 2

    started = time.perf_counter()
    if args.retrieval_only:
        from src.query.hybrid_search import HybridSearcher

        searcher = HybridSearcher()
        records = [r for r in (evaluate_retrieval(c, searcher) for c in cases) if r is not None]
    else:
        from src.query.pipeline import RagPipeline

        records = run(cases, RagPipeline(), use_judge=not args.no_judge, workers=args.workers)
    meta = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": config.LLM_MODEL,
            "embedding_model": config.EMBED_MODEL, "top_k": config.TOP_K, "judge": not (args.no_judge or args.retrieval_only),
            "mode": "retrieval-only" if args.retrieval_only else "full",
            "questions": len(records), "wall_seconds": round(time.perf_counter() - started, 1)}
    report = build_report(records, meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print_report(report)
    print(f"\n  results written to {args.out}  ({meta['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
