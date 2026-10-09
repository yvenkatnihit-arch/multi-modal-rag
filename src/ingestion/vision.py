"""Gemini vision: turn an image into searchable text (caption, OCR text, chart/diagram details).

Also reused later for scanned PDF pages (OCR fallback) and images embedded in PDFs.
"""
import hashlib
import json
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from src import config
from src.core.gemini_client import call_with_retries, get_client
from src.core.images import prepare_image

PROMPT_VERSION = 2         # bump when the prompt/schema changes: cached analyses are then redone

PROMPT = """You are indexing an image for a search system. Describe only what is actually visible; \
never guess or invent details. Fill these fields:
- image_type: photo, chart, diagram, document, screenshot, or other.
- caption: one or two sentences saying what the image shows. For a product photo give the item, colour, \
style and who it is for if evident. For a document or receipt say what kind it is.
- text: ALL text visible in the image, transcribed exactly as printed. Put every printed line on its own \
line, separated by a newline character, never run lines together. Keep numbers, dates and currency \
symbols exactly. Use an empty string if there is no text.
- details: for a chart, give the title, axis labels, categories or series, and the values or trend you can \
read. For a diagram, list the components and how they connect. Otherwise use an empty string."""


class ImageAnalysis(BaseModel):
    image_type: Literal["photo", "chart", "diagram", "document", "screenshot", "other"]
    caption: str = Field(description="what the image shows")
    text: str = Field(default="", description="all visible text, transcribed exactly")
    details: str = Field(default="", description="chart or diagram description")


def analyze_image(data: bytes, mime: str) -> ImageAnalysis:
    from google.genai import types

    def call():
        return get_client().models.generate_content(
            model=config.LLM_MODEL,
            contents=[types.Part.from_bytes(data=data, mime_type=mime), PROMPT],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ImageAnalysis,
                thinking_config=types.ThinkingConfig(thinking_budget=0),   # extraction, not reasoning
            ),
        )

    resp = call_with_retries(call, "vision call")
    if resp.parsed is None:
        raise ValueError(f"model returned no analysis (blocked or malformed): {(resp.text or '')[:120]!r}")
    return resp.parsed


def _load_cached(sidecar: Path, digest: str) -> ImageAnalysis | None:
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if (data["sha1"], data["model"], data["prompt_version"]) == (digest, config.LLM_MODEL, PROMPT_VERSION):
            return ImageAnalysis(**data["analysis"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def analyze_with_cache(raw: bytes, sidecar: Path, analyzer: Callable[[bytes, str], ImageAnalysis] | None = None) -> ImageAnalysis:
    """Analyze image bytes, reusing a saved answer when the bytes, the model and the prompt version are all unchanged.
    Shared by the image parser (photos) and the PDF parser (scanned pages), so the cache rules live in one place."""
    digest = hashlib.sha1(raw).hexdigest()
    cached = _load_cached(sidecar, digest)
    if cached is not None:
        return cached
    data, mime = prepare_image(raw)                       # ValueError for corrupt / tiny images
    analysis = (analyzer or analyze_image)(data, mime)
    sidecar.write_text(json.dumps({
        "sha1": digest, "model": config.LLM_MODEL, "prompt_version": PROMPT_VERSION,
        "analysis": analysis.model_dump(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return analysis


def analysis_to_text(a: ImageAnalysis, name: str, noun: str = "Image", text_label: str = "Text in image") -> str:
    """The searchable text for an image (or a scanned page): empty sections are left out."""
    parts = [f"{noun} {name} ({a.image_type}).", a.caption.strip()]
    if a.text.strip():
        parts.append(f"{text_label}:\n" + a.text.strip())
    if a.details.strip():
        parts.append("Chart/diagram details: " + a.details.strip())
    return "\n".join(p for p in parts if p)
