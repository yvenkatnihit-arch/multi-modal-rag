"""Table parser: .csv .xlsx .json (list of records)  ->  table Records + a full copy in assets/.

For every table it emits
  * one "summary" record: shape, per-column statistics, first rows  (makes the table findable)
  * "rows-a-b" records: rows as text with the header repeated      (makes single rows findable)
    - small tables: every row;  big tables: a fixed random sample, and the summary says so
and saves the COMPLETE table as Parquet so table_query can compute exact answers.
"""
import csv
import json
import logging
import math
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype, is_object_dtype

from src import config
from src.core.assets import safe_name, save_table
from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.common import clean_text, read_text_file
from src.core.records import Metadata, Record, pdf_table_ref
from src.core.table_types import is_amount_text as _is_amount_text  # one shared definition (also used by table_query)
from src.core.table_types import is_date_like as _is_date_like
from src.core.table_types import to_number  # noqa: F401  (re-exported: tests and callers import it from here)

log = logging.getLogger(__name__)

MAX_SUMMARY_COLUMNS = 60
MAX_COLUMN_LINE = 400


# ------------------------------------------------------------------ loading
def _read_csv(path) -> pd.DataFrame:
    with open(path, "rb") as f:                       # only the first 8 KB is needed to detect the delimiter
        sample = f.read(8192).decode("utf-8-sig", errors="replace")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path.name}")


def _records_from_json(data) -> list[dict] | None:
    def is_records(v):
        return isinstance(v, list) and v and all(isinstance(x, dict) for x in v[:20])
    if is_records(data):
        return data
    if isinstance(data, dict):
        lists = [v for v in data.values() if is_records(v)]
        if lists:
            return max(lists, key=len)       # the biggest list of objects is the table
    return None


def _read_json(path) -> pd.DataFrame:
    records = _records_from_json(json.loads(read_text_file(path)))
    if records is None:
        raise ValueError("JSON has no list of records")
    df = pd.json_normalize(records, sep=".")
    first = {c.split(".", 1)[0] for c in df.columns if "." in c}
    if len(first) == 1 and all(c.startswith(next(iter(first)) + ".") for c in df.columns):
        prefix = next(iter(first)) + "."                  # a single wrapper key like "restaurant."
        df.columns = [c[len(prefix):] for c in df.columns]
    return df


def _load_tables(routed: RoutedFile) -> list[tuple[str, pd.DataFrame]]:
    ext = routed.path.suffix.lower()
    if ext == ".xlsx":
        sheets = pd.read_excel(routed.path, sheet_name=None)
        out, used = [], set()
        for name, df in sheets.items():
            tid = safe_name(str(name))
            while tid in used:
                tid += "_"
            used.add(tid)
            out.append((tid, df))
        return out
    return [("main", _read_json(routed.path) if ext == ".json" else _read_csv(routed.path))]


# ------------------------------------------------------------------ cleaning
def _to_text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def _unique(names: list[str]) -> list[str]:
    """['a', 'a', 'b'] -> ['a', 'a_2', 'b']"""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _unique([str(c).strip() or "column" for c in df.columns])     # Parquet rejects duplicate names
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for c in [c for c in df.columns if c.startswith("Unnamed:")]:      # leftover index columns
        s = df[c]
        if is_integer_dtype(s) and (s.reset_index(drop=True) == range(len(s))).all() or \
           is_integer_dtype(s) and (s.reset_index(drop=True) == range(1, len(s) + 1)).all():
            df = df.drop(columns=c)
    for c in df.columns:
        if is_object_dtype(df[c]):                                      # mixed types would break Parquet
            df[c] = df[c].map(_to_text)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ formatting
def _num(x) -> str:
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:.0f}" if abs(x) >= 1e4 else f"{round(x, 3):g}"


def _cell(v, maxlen: int) -> str:
    if v is None or (not isinstance(v, (str, list, dict)) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        v = int(v)
    elif isinstance(v, float):
        v = round(v, 4)
    s = " ".join(str(v).replace("|", "/").split())
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def _counts(values: pd.Series, limit: int) -> str:
    return ", ".join(f"{v} ({c:,})" for v, c in values.value_counts().head(limit).items())


def _describe_column(name: str, s: pd.Series) -> str:
    nn = s.dropna()
    missing = int(s.isna().sum())
    tail = f"; {missing:,} missing" if missing else ""
    if nn.empty:
        return f"- {name} (empty): all values missing"
    if is_bool_dtype(s):
        line = f"- {name} (true/false): {_counts(nn, 5)}"
    elif is_numeric_dtype(s):
        kind = "integer" if is_integer_dtype(s) or (nn % 1 == 0).all() else "decimal"
        line = (f"- {name} ({kind}): min {_num(nn.min())}, max {_num(nn.max())}, "
                f"mean {_num(nn.mean())}, median {_num(nn.median())}")
        if nn.nunique() <= 10:
            line += "; values: " + _counts(nn, 10)
    elif _is_amount_text(nn):
        num = to_number(nn).dropna()
        line = (f"- {name} (number stored as text, e.g. \"{_cell(nn.iloc[0], 20)}\"; clean symbols before arithmetic): "
                f"min {_num(num.min())}, max {_num(num.max())}, mean {_num(num.mean())}, median {_num(num.median())}")
    elif _is_date_like(nn):
        d = pd.to_datetime(nn.astype(str), errors="coerce", utc=True, format="mixed").dropna()
        line = f"- {name} (date): from {d.min().date()} to {d.max().date()}"
    else:
        text = nn.astype(str)
        nunique = text.nunique()
        if nunique <= config.TABLE_ENUM_MAX:
            line = f"- {name} (text): values: {_counts(text, config.TABLE_ENUM_MAX)}"
        elif text.str.len().mean() > 60:
            line = f"- {name} (free text): avg {text.str.len().mean():.0f} chars, e.g. \"{_cell(text.iloc[0], 80)}\""
        else:
            line = f"- {name} (text): {nunique:,} unique values; most common: {_counts(text, 5)}"
    line += tail
    return line if len(line) <= MAX_COLUMN_LINE else line[: MAX_COLUMN_LINE - 1] + "…"


def _render_rows(df: pd.DataFrame, row_numbers: list[int], cell_max: int) -> list[str]:
    return [f"{n} | " + " | ".join(_cell(v, cell_max) for v in df.iloc[n - 1]) for n in row_numbers]


def _header_line(df: pd.DataFrame) -> str:
    return "row | " + " | ".join(str(c)[:40] for c in df.columns)


# ------------------------------------------------------------------ records
def _title(source: str, table_id: str) -> str:
    pdf = pdf_table_ref(table_id)
    if pdf:
        return f"Table {source} (page {pdf[0]}, table {pdf[1]})"
    return f"Table {source}" + ("" if table_id == "main" else f" [sheet {table_id}]")


def _sampled_rows(n_rows: int) -> list[int]:
    if n_rows <= config.TABLE_EMBED_ROWS:
        return list(range(1, n_rows + 1))
    rng = np.random.default_rng(config.TABLE_SAMPLE_SEED)
    return sorted(int(i) + 1 for i in rng.choice(n_rows, config.TABLE_EMBED_ROWS, replace=False))


def _summary_text(df: pd.DataFrame, source: str, topic_id: str, table_id: str,
                  embedded: int, cell_max: int) -> str:
    n_rows, n_cols = df.shape
    lines = [f"{_title(source, table_id)} (topic {topic_id}): {n_rows:,} rows x {n_cols} columns.", "Columns:"]
    cols = list(df.columns)
    lines += [_describe_column(c, df[c]) for c in cols[:MAX_SUMMARY_COLUMNS]]
    if len(cols) > MAX_SUMMARY_COLUMNS:
        lines.append(f"- ... and {len(cols) - MAX_SUMMARY_COLUMNS} more columns")
    lines.append("First rows:")
    lines.append(_header_line(df))
    lines += _render_rows(df, list(range(1, min(3, n_rows) + 1)), cell_max)
    if embedded < n_rows:
        lines.append(f"Note: only a sample of {embedded:,} of {n_rows:,} rows is searchable as text; "
                     "counts, sums and averages are computed over all rows with the table calculator.")
    return "\n".join(lines)


def _row_groups(df: pd.DataFrame, rows: list[int], cell_max: int) -> list[tuple[int, int, list[str]]]:
    """Pack rendered rows into groups of roughly TABLE_GROUP_CHARS characters."""
    groups, current, size = [], [], 0
    for n, line in zip(rows, _render_rows(df, rows, cell_max)):
        if current and size + len(line) > config.TABLE_GROUP_CHARS:
            groups.append((current[0][0], current[-1][0], [l for _, l in current]))
            current, size = [], 0
        current.append((n, line))
        size += len(line) + 1
    if current:
        groups.append((current[0][0], current[-1][0], [l for _, l in current]))
    return groups


def table_records(df: pd.DataFrame, source: str, topic_id: str, table_id: str, page: int | None = None) -> list[Record]:
    """The searchable records for ONE table, and the full table saved in assets/ for exact calculations.
    Used for CSV/Excel/JSON files and for tables found inside PDFs (which pass their page number)."""
    df = _clean(df)
    if df.empty:
        log.warning("[%s] %s [%s]: no data rows; skipped", topic_id, source, table_id)
        return []
    save_table(df, topic_id, source, table_id)

    cell_max = max(20, min(config.TABLE_CELL_MAX, 900 // len(df.columns)))
    rows = _sampled_rows(len(df))
    meta = lambda part: Metadata(topic_id, source, "table", page=page, table_id=table_id, part=part)

    records = [Record(clean_text(_summary_text(df, source, topic_id, table_id, len(rows), cell_max)), meta("summary"))]
    header = _header_line(df)
    kind = "sample" if len(rows) < len(df) else "rows"       # a sample's row numbers are not contiguous
    for first, last, lines in _row_groups(df, rows, cell_max):
        what = f"sampled rows {first}-{last}" if kind == "sample" else f"rows {first}-{last}"
        text = "\n".join([f"{_title(source, table_id)}, {what}:", header, *lines])
        records.append(Record(text, meta(f"{kind}-{first}-{last}")))
    return records


def parse_table(routed: RoutedFile, topic_id: str) -> list[Record]:
    records: list[Record] = []
    for table_id, raw in _load_tables(routed):
        records += table_records(raw, routed.source, topic_id, table_id)
    return records
