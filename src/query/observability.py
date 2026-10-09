"""Per-question tracing: how long each step took, what it decided, and how many tokens the LLM calls used.

    trace = Trace(question)
    with trace.step("route") as info:       # times the block; whatever is put in `info` is recorded
        info["topics"] = [...]
    trace.finish(llm_usages)
    QueryLog().write(...)                    # one JSON line per question in logs/queries.jsonl

Logging must never break an answer: write failures are swallowed (and logged as a warning).
"""
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src import config
from src.core.llm import Usage

log = logging.getLogger(__name__)


@dataclass
class Step:
    name: str
    seconds: float
    info: dict = field(default_factory=dict)


class Trace:
    def __init__(self, question: str):
        self.question = question
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        self.steps: list[Step] = []
        self.total_seconds = 0.0
        self.llm_calls = 0
        self.tokens = {"prompt": 0, "output": 0, "thinking": 0}

    @contextmanager
    def step(self, name: str):
        info: dict = {}
        t = time.perf_counter()
        try:
            yield info
        except Exception as e:                              # the step is still recorded, with what went wrong
            info["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.steps.append(Step(name, time.perf_counter() - t, info))

    def get(self, name: str) -> Step | None:
        return next((s for s in self.steps if s.name == name), None)

    def finish(self, usages: Sequence[Usage]) -> None:
        self.total_seconds = time.perf_counter() - self._t0
        self.llm_calls = len(usages)
        self.tokens = {
            "prompt": sum(u.prompt_tokens for u in usages),
            "output": sum(u.output_tokens for u in usages),
            "thinking": sum(u.thinking_tokens for u in usages),
        }

    def to_dict(self) -> dict:
        d = {
            "started_at": self.started_at, "total_seconds": round(self.total_seconds, 3),
            "llm_calls": self.llm_calls, "tokens": self.tokens,
            "steps": [{"name": s.name, "seconds": round(s.seconds, 3), **s.info} for s in self.steps],
        }
        return json.loads(json.dumps(d, default=str))      # guarantee it is JSON-safe


class QueryLog:
    """Appends one JSON object per question to logs/queries.jsonl."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config.LOGS_DIR / "queries.jsonl"

    def write(self, record: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            log.warning("could not write the query log (%s)", e)


# ---------------------------------------------------------------- human-readable trace
def step_summary(step: Step) -> str:
    """One readable line about what a step did (used by the terminal trace and the Streamlit trace panel)."""
    i, n = step.info, step.name
    if "error" in i and n not in ("rewrite",):
        return f"FAILED: {i['error']}"
    if n == "rewrite":
        if not i.get("used_llm"):
            return "skipped (no history)"
        base = f"{'rewritten' if i.get('changed') else 'unchanged'}: {i.get('standalone')!r}"
        return base + (f"  [kept original: {i['error']}]" if i.get("error") else "")
    if n == "route":
        return f"{i.get('topics')}" + ("  (router unavailable: searching all topics)" if i.get("fallback") else "")
    if n == "search":
        parts = [f"{t}: {len(h)} hits" for t, h in i.get("per_topic", {}).items()]
        return "; ".join(parts) + (f"  (failed: {list(i['errors'])})" if i.get("errors") else "")
    if n == "merge":
        return f"{len(i.get('final', []))} kept, {len(i.get('duplicates', []))} duplicates removed"
    if n == "context":
        return f"{len(i.get('items', []))} chunks, ~{i.get('tokens')} tokens" + (f", {len(i['dropped'])} dropped" if i.get("dropped") else "")
    if n == "generate":
        calcs = i.get("calculations") or []
        extra = f", {len(calcs)} table calculation(s)" if calcs else ""
        return (f"answered, used {i.get('used_chunks')}" if i.get("answerable") else f"refused ({i.get('refusal_reason')})") + extra
    if n == "citations":
        return f"{len(i.get('sources', []))} source(s)" + ("  !! uncited answer" if i.get("uncited") else "")
    return ""


def format_trace(trace: Trace) -> str:
    t = trace.tokens
    lines = [f"trace: {trace.total_seconds:.1f}s total | {trace.llm_calls} LLM call(s) | "
             f"tokens in {t['prompt']} / out {t['output']} / thinking {t['thinking']}"]
    lines += [f"  {s.name:<10}{s.seconds:5.2f}s  {step_summary(s)}" for s in trace.steps]
    return "\n".join(lines)
