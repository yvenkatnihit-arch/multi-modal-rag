"""Preparing an image to be sent to the model: fix rotation, cap the size, re-encode.

Shared by both pipelines: ingestion sends photos, scanned pages and PDF figures for description; the query side sends the
original pictures to the answer step. Kept here so neither pipeline has to import the other.
"""
import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 2000             # longest side sent to the model (cost vs. legibility of small print)
MIN_SIDE = 16


def prepare_image(path_or_bytes) -> tuple[bytes, str]:
    """Open, fix rotation, cap the size, re-encode. Returns (bytes, mime). Raises ValueError for bad images."""
    try:
        src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
        img = ImageOps.exif_transpose(Image.open(src))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"not a readable image: {e}") from e
    if min(img.size) < MIN_SIDE:
        raise ValueError(f"image too small ({img.size[0]}x{img.size[1]})")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        img.convert("RGBA").save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"
