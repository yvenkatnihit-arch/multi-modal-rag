"""Where parsed originals live: assets/<topic>/tables/*.parquet and assets/<topic>/images/*.

The vector store holds searchable text *about* a table; the full table sits here untouched so table_query can
compute exact answers with pandas. A small manifest (assets/<topic>/tables/manifest.json) records what each saved
table is (source file, sheet, row count, column types): it is what lets the answer step know which tables exist.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

from src import config
from src.core.table_types import column_kind

MANIFEST = "manifest.json"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "x"


def table_asset_path(topic_id: str, source: str, table_id: str) -> Path:
    stem = safe_name(source.replace("/", "__"))
    return config.ASSETS_DIR / topic_id / "tables" / f"{stem}.{safe_name(table_id)}.parquet"


def image_asset_path(topic_id: str, source: str) -> Path:
    return config.ASSETS_DIR / topic_id / "images" / safe_name(source.replace("/", "__"))


def page_asset_path(topic_id: str, source: str, page: int) -> Path:
    """The rendered image of one scanned PDF page (kept so the answer step can show the actual page)."""
    return config.ASSETS_DIR / topic_id / "pages" / f"{safe_name(source.replace('/', '__'))}.p{page}.jpg"


def pdf_image_asset_path(topic_id: str, source: str, page: int, n: int, ext: str) -> Path:
    """A picture extracted from a PDF page: the n-th figure on that page."""
    return config.ASSETS_DIR / topic_id / "pdf_images" / f"{safe_name(source.replace('/', '__'))}.p{page}i{n}.{ext}"


def relative_to_root(path: Path) -> str:
    """Stored in chunk metadata, so the project can be moved without breaking image links."""
    return path.resolve().relative_to(config.ROOT.resolve()).as_posix()


def resolve_asset(rel_path: str) -> Path:
    return config.ROOT / rel_path


# ------------------------------------------------------------------ manifest
def _manifest_path(topic_id: str) -> Path:
    return config.ASSETS_DIR / topic_id / "tables" / MANIFEST


def _read_manifest(topic_id: str) -> dict:
    try:
        return json.loads(_manifest_path(topic_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_manifest(topic_id: str, data: dict) -> None:
    path = _manifest_path(topic_id)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)                                    # atomic: a reader never sees a half-written file


def save_table(df: pd.DataFrame, topic_id: str, source: str, table_id: str) -> Path:
    path = table_asset_path(topic_id, source, table_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    manifest = _read_manifest(topic_id)
    manifest[path.name] = {
        "source": source, "table_id": table_id, "rows": int(len(df)),
        "columns": {str(c): column_kind(df[c]) for c in df.columns},
    }
    _write_manifest(topic_id, manifest)
    return path


def load_table(topic_id: str, source: str, table_id: str) -> pd.DataFrame:
    path = table_asset_path(topic_id, source, table_id)
    if not path.exists():
        raise FileNotFoundError(f"no stored table for {topic_id}/{source} [{table_id}]; run ingestion first")
    return pd.read_parquet(path)


def list_tables(topic_id: str) -> list[dict]:
    """Saved tables of a topic: [{file, source, table_id, rows, columns: {name: kind}}], skipping vanished files."""
    folder = _manifest_path(topic_id).parent
    out = [{"file": name, **entry} for name, entry in _read_manifest(topic_id).items() if (folder / name).exists()]
    return sorted(out, key=lambda e: (e["source"], e["table_id"]))


def prune_tables(topic_id: str, keep: set[tuple[str, str]]) -> list[str]:
    """Delete saved tables (and manifest entries) whose (source, table_id) is not in `keep`, plus any parquet file the
    manifest does not know. Called after a clean ingestion, so a deleted source file cannot keep answering questions."""
    folder = _manifest_path(topic_id).parent
    if not folder.exists():
        return []
    manifest = _read_manifest(topic_id)
    removed = []
    for name, entry in list(manifest.items()):
        if (entry["source"], entry["table_id"]) not in keep:
            (folder / name).unlink(missing_ok=True)
            del manifest[name]
            removed.append(name)
    for f in folder.glob("*.parquet"):
        if f.name not in manifest:
            f.unlink()
            removed.append(f.name)
    _write_manifest(topic_id, manifest)
    return removed
