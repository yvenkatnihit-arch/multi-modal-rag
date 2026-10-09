"""Helpers shared by all parsers."""
import re
import unicodedata
from pathlib import Path

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise text for embedding and display.

    NFKC folds ligatures and look-alikes ('ﬁve' -> 'five', non-breaking space -> space).
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def read_text_file(path: Path) -> str:
    """Decode bytes as UTF-8 (BOM tolerated), falling back to Windows-1252 then Latin-1."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")
