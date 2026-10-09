"""Pictures for the answer step: which original images (photos, PDF figures, scanned pages) behind the retrieved chunks
should the model actually look at, loaded and ready to send.

Chunk text is only a description of a picture; the picture is the ground truth. Selection is best-first (the context
is already in rank order), one picture per file, capped to keep cost and latency bounded.
"""
import logging
from dataclasses import dataclass

from src import config
from src.core.assets import resolve_asset
from src.query.context import Context
from src.core.images import prepare_image

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Picture:
    number: int            # the context chunk it belongs to: the [n] the model may cite
    label: str             # e.g. "image X51005361900.jpg"
    path: str              # image_path from the chunk metadata
    data: bytes
    mime: str

    @property
    def caption(self) -> str:
        return f"Picture for chunk [{self.number}] ({self.label}):"

    def as_llm_input(self) -> tuple[str, bytes, str]:
        return self.caption, self.data, self.mime


def select_pictures(context: Context, max_n: int | None = None) -> list[Picture]:
    """Load up to max_n pictures for the context's chunks. A file that is missing or unreadable is skipped, not fatal."""
    limit = config.MAX_PICTURES_TO_LLM if max_n is None else max_n
    out: list[Picture] = []
    seen: set[str] = set()
    for item in context.items:
        if len(out) >= limit:
            break
        path = item.image_path
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            data, mime = prepare_image(resolve_asset(path).read_bytes())
        except (OSError, ValueError) as e:
            log.warning("picture %s for chunk [%d] could not be loaded (%s); answering from its description", path, item.number, e)
            continue
        out.append(Picture(item.number, item.label, path, data, mime))
    return out
