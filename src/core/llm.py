"""Structured text-only LLM calls (rewriter, router, answer generator)."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel

from src import config
from src.core.gemini_client import call_with_retries, get_client


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens, self.output_tokens + other.output_tokens,
                     self.thinking_tokens + other.thinking_tokens)


_tracker: ContextVar[list | None] = ContextVar("llm_usage", default=None)


@contextmanager
def track_usage():
    """Everything LLM-related called inside this block has its token usage appended to the yielded list,
    so the pipeline can total up router + rewriter + generator without changing their signatures."""
    calls: list[Usage] = []
    token = _tracker.set(calls)
    try:
        yield calls
    finally:
        _tracker.reset(token)


def record_usage(usage: Usage) -> None:
    calls = _tracker.get()
    if calls is not None:
        calls.append(usage)


def generate_structured_ex(
    prompt: str,
    schema: type[BaseModel],
    system: str | None = None,
    thinking_budget: int = 0,
    images: Sequence[tuple[str, bytes, str]] | None = None,
) -> tuple[BaseModel, Usage]:
    """Ask Gemini for JSON matching `schema`. Returns (parsed answer, token usage).
    `images` are (caption, bytes, mime) triples: each picture is sent right after its caption, before the prompt.
    Raises ValueError if the answer is blocked or malformed."""
    from google.genai import types

    contents: str | list = prompt
    if images:
        contents = []
        for caption, data, mime in images:
            contents += [caption, types.Part.from_bytes(data=data, mime_type=mime)]
        contents.append(prompt)

    def call():
        return get_client().models.generate_content(
            model=config.LLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            ),
        )

    resp = call_with_retries(call, "LLM call")
    if resp.parsed is None:
        raise ValueError(f"model returned no structured answer (blocked or malformed): {(resp.text or '')[:120]!r}")
    u = resp.usage_metadata
    usage = Usage(
        prompt_tokens=getattr(u, "prompt_token_count", 0) or 0,
        output_tokens=getattr(u, "candidates_token_count", 0) or 0,
        thinking_tokens=getattr(u, "thoughts_token_count", 0) or 0,
    )
    record_usage(usage)
    return resp.parsed, usage


def generate_structured(prompt: str, schema: type[BaseModel], system: str | None = None) -> BaseModel:
    """Quick decisions (routing, rewriting): no thinking, usage not needed."""
    return generate_structured_ex(prompt, schema, system, thinking_budget=0)[0]
