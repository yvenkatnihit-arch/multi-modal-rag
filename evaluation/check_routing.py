"""Measures the router and the query rewriter against evaluation/routing_cases.json with the real LLM.

    python -m evaluation.check_routing

Costs about two dozen small Gemini calls. Not part of pytest (needs the network).
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402,F401  (first: installs the gRPC telemetry stub before chromadb is imported)
from src.query.query_rewriter import rewrite_question  # noqa: E402
from src.query.router import TopicRouter  # noqa: E402

CASES = Path(__file__).resolve().parent / "routing_cases.json"


def main() -> int:
    logging.disable(logging.CRITICAL)
    cases = json.loads(CASES.read_text(encoding="utf-8"))

    router = TopicRouter()
    ok = 0
    print("ROUTER")
    for c in cases["routing"]:
        d = router.route(c["question"])
        good = set(d.topics) == set(c["expected"])
        ok += good
        note = " (fallback)" if d.fallback else ""
        print(f"  {'ok  ' if good else 'MISS'} {c['question'][:62]:62} -> {d.topics}{note}"
              + ("" if good else f"   expected {c['expected']}  | {d.reason}"))
    print(f"  routing accuracy: {ok}/{len(cases['routing'])}\n")

    print("REWRITER")
    ok_rewrite = 0
    for c in cases["rewriting"]:
        r = rewrite_question(c["question"], c["history"])
        if c.get("unchanged"):
            good = r.question == c["question"]
        else:
            good = all(s.lower() in r.question.lower() for s in c["must_contain"])
        ok_rewrite += good
        print(f"  {'ok  ' if good else 'MISS'} {c['question']!r} -> {r.question!r}" + (f"  [{r.error}]" if r.error else ""))
    print(f"  rewrite accuracy: {ok_rewrite}/{len(cases['rewriting'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
