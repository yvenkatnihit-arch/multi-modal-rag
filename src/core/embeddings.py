"""Gemini embeddings: batched, retried, with the right task type for documents vs queries."""
from src import config
from src.core.gemini_client import call_with_retries, get_client

BATCH_SIZE = 100          # API limit per embed_content call


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    from google.genai import types

    def call():
        resp = get_client().models.embed_content(
            model=config.EMBED_MODEL, contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors = [list(e.values) for e in resp.embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    return call_with_retries(call, "embedding call")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for storage."""
    if any(not t.strip() for t in texts):
        raise ValueError("cannot embed empty text")
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        out += _embed_batch(texts[i : i + BATCH_SIZE], "RETRIEVAL_DOCUMENT")
    return out


def embed_query(text: str) -> list[float]:
    """Embed a user question for searching."""
    if not text.strip():
        raise ValueError("cannot embed an empty query")
    return _embed_batch([text], "RETRIEVAL_QUERY")[0]
