"""table_query: exact calculations over the FULL stored tables (assets/), not over the sampled rows in the index.

The model never writes code. It fills in a TableRequest (table, operation, column, filters, group-by, order, limit)
and THIS module turns that into pandas. Every column name and operation is validated; nothing is eval'd.

Honesty rules:
  * a filter that matches no rows is reported as such (never a silent 0 or a made-up mean), with the closest real
    values suggested, which also copes with typos and the garbled text in the Zomato data ('São Paulo' -> 'Sí£o Paulo')
  * numbers stored as text ('$6.60') and date strings are converted the same way the parser described them
  * values that cannot be used are counted and reported in `notes`, not dropped silently
"""
import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from src.core import assets
from src.core.records import pdf_table_ref
from src.core.table_types import column_kind, normalize_one, normalize_text, to_number

log = logging.getLogger(__name__)

MAX_REQUESTS = 3
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MAX_GROUP_COLUMNS = 2
MAX_CATALOG_COLUMNS = 60

Operation = Literal["count", "sum", "mean", "median", "min", "max", "nunique"]
FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "in", "is_null", "not_null"]

_OP_WORDS = {"count": "count of", "sum": "sum of", "mean": "average (mean) of", "median": "median of",
             "min": "minimum of", "max": "maximum of", "nunique": "number of distinct values of"}
_KIND_WORDS = {"amount_text": "number stored as text", "empty": "empty", "boolean": "true/false"}


# ------------------------------------------------------------------ what the model may ask for
class Filter(BaseModel):
    column: str
    op: FilterOp
    value: str | None = Field(default=None, description="the value to compare with, as text (e.g. '4.5', 'New Delhi')")
    values: list[str] = Field(default_factory=list, description="for op 'in': the allowed values")


class TableRequest(BaseModel):
    table: str = Field(description="table reference from the <tables> list, e.g. 'T1'")
    operation: Operation
    column: str | None = Field(default=None, description="the column to aggregate; may be omitted for 'count' of rows")
    filters: list[Filter] = Field(default_factory=list, description="all must hold (AND)")
    group_by: list[str] = Field(default_factory=list, description="up to two columns to group by")
    order: Literal["desc", "asc"] | None = Field(default=None, description="for grouped results; default desc")
    limit: int | None = Field(default=None, description="for grouped results: keep this many groups (default 20, max 50)")


# ------------------------------------------------------------------ the catalog of what can be queried
@dataclass(frozen=True)
class TableInfo:
    ref: str                       # "T1": what the model uses to point at the table
    topic_id: str
    source: str
    table_id: str
    rows: int
    columns: dict[str, str]        # column name -> kind (integer, decimal, amount_text, date, text, boolean)

    @property
    def title(self) -> str:
        pdf = pdf_table_ref(self.table_id)
        where = "" if self.table_id == "main" else (f" [page {pdf[0]}, table {pdf[1]}]" if pdf else f" [sheet {self.table_id}]")
        return f"{self.topic_id}/{self.source}{where}"


def build_catalog(topic_ids: Sequence[str], lister: Callable[[str], list[dict]] = assets.list_tables) -> list[TableInfo]:
    infos: list[TableInfo] = []
    for topic in topic_ids:
        for e in lister(topic):
            infos.append(TableInfo(f"T{len(infos) + 1}", topic, e["source"], e["table_id"], e["rows"], e["columns"]))
    return infos


def catalog_text(tables: Sequence[TableInfo]) -> str:
    lines = []
    for t in tables:
        cols = [f"{name} ({_KIND_WORDS.get(kind, kind)})" for name, kind in list(t.columns.items())[:MAX_CATALOG_COLUMNS]]
        more = f", ... and {len(t.columns) - MAX_CATALOG_COLUMNS} more" if len(t.columns) > MAX_CATALOG_COLUMNS else ""
        lines.append(f"{t.ref}: {t.title}, {t.rows:,} rows. Columns: {', '.join(cols)}{more}")
    return "\n".join(lines)


# ------------------------------------------------------------------ results
@dataclass(frozen=True)
class TableResult:
    ref: str                                       # "C1": how the answer cites this calculation
    table_ref: str
    topic_id: str
    source: str
    table_id: str
    description: str
    ok: bool
    value: float | int | str | None = None         # a single figure ...
    rows: list[dict] | None = None                 # ... or grouped figures
    rows_note: str | None = None                   # how the groups are ordered and whether some were cut off
    rows_total: int = 0
    rows_matched: int = 0
    notes: list[str] = field(default_factory=list)
    suggestions: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None

    @property
    def number(self) -> int:
        return int(self.ref[1:])

    def to_text(self) -> str:
        head = f"{self.ref} [{self.table_ref}: {self.topic_id}/{self.source}] {self.description}"
        lines = [head]
        if not self.ok:
            lines.append(f"   ERROR: {self.error}")
        elif self.rows is not None:
            if self.rows:
                lines.append(f"   groups ({self.rows_note}):")
                for r in self.rows:
                    keys = ", ".join(f"{k} = {v}" for k, v in r.items() if k != "value")
                    lines.append(f"   {keys} -> {r['value']}")
            else:
                lines.append("   no groups (no rows matched)")
        elif self.value is None:
            lines.append("   result: none (no rows matched, so there is nothing to compute)")
        else:
            lines.append(f"   result: {self.value}")
        if self.ok:
            lines.append(f"   computed over {self.rows_matched:,} of {self.rows_total:,} rows of the full table")
        lines += [f"   note: {n}" for n in self.notes]
        for col, vals in self.suggestions.items():
            lines.append(f"   closest existing values in '{col}': {', '.join(repr(v) for v in vals)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"ref": self.ref, "table": f"{self.topic_id}/{self.source}", "description": self.description,
                "ok": self.ok, "value": self.value, "rows": self.rows, "rows_matched": self.rows_matched,
                "rows_total": self.rows_total, "notes": self.notes, "suggestions": self.suggestions, "error": self.error}


class QueryError(ValueError):
    """A request that cannot be run; the message (with suggestions) goes back to the model."""

    def __init__(self, message: str, suggestions: dict[str, list[str]] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or {}


# ------------------------------------------------------------------ helpers
def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_column(name: str, columns: Sequence[str]) -> str:
    """Exact name, else case/punctuation-insensitive match, else a QueryError listing the closest columns."""
    if name in columns:
        return name
    by_key: dict[str, list[str]] = {}
    for c in columns:
        by_key.setdefault(_key(c), []).append(c)
    hit = by_key.get(_key(name))
    if hit and len(hit) == 1:
        return hit[0]
    close = difflib.get_close_matches(name, list(columns), n=3, cutoff=0.4) or difflib.get_close_matches(_key(name), list(by_key), n=3, cutoff=0.4)
    close = [by_key[c][0] if c in by_key else c for c in close]
    raise QueryError(f"there is no column named '{name}'", {"columns": close or list(columns)[:5]})


def _native(x):
    """numpy -> plain Python, floats rounded for readability."""
    if x is None or (not isinstance(x, (str, list, dict)) and pd.isna(x)):      # NaN, NaT, <NA>
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float):
        return int(x) if x.is_integer() and abs(x) < 1e15 else round(x, 6)
    return x


def _quote(v) -> str:
    return repr(v)


def _numeric(s: pd.Series, kind: str, column: str, notes: list[str]) -> pd.Series:
    if kind == "amount_text":
        num = to_number(s)
    elif kind in ("integer", "decimal"):
        num = pd.to_numeric(s, errors="coerce")
    else:
        raise QueryError(f"column '{column}' holds {_KIND_WORDS.get(kind, kind)} values, so sum/mean/median/min/max cannot be computed on it")
    ignored = int(s.notna().sum() - num.notna().sum())
    if ignored:
        notes.append(f"{ignored:,} value(s) in '{column}' were not numeric and were ignored")
    return num.astype("float64")


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string"), errors="coerce", utc=True, format="mixed")


def _need_value(f: Filter) -> str:
    if f.value is None or not str(f.value).strip():
        raise QueryError(f"filter on '{f.column}' with '{f.op}' needs a value")
    return str(f.value).strip()


def _filter_mask(df: pd.DataFrame, f: Filter, col: str, kind: str) -> pd.Series:
    s = df[col]
    if f.op in ("is_null", "not_null"):
        empty = s.isna() | (s.astype("string").str.strip() == "") if kind == "text" else s.isna()
        return empty if f.op == "is_null" else ~empty

    if kind in ("integer", "decimal", "amount_text"):
        num = to_number(s) if kind == "amount_text" else pd.to_numeric(s, errors="coerce")
        if f.op == "contains":
            raise QueryError(f"'contains' only works on text columns, and '{col}' is numeric")
        wanted = f.values if f.op == "in" and f.values else [_need_value(f)]
        nums = to_number(pd.Series(wanted)).tolist()
        if any(pd.isna(n) for n in nums):
            raise QueryError(f"filter value {wanted} is not a number, but column '{col}' is numeric")
        if f.op == "in":
            return num.isin(nums)
        v = nums[0]
        return {"==": num == v, "!=": (num != v) & num.notna(), ">": num > v, ">=": num >= v, "<": num < v, "<=": num <= v}[f.op]

    if kind == "date":
        if f.op in ("contains", "in"):
            raise QueryError(f"'{f.op}' is not supported on the date column '{col}'; use ==, !=, >, >=, <, <=")
        raw = _need_value(f)
        v = _dates(pd.Series([raw])).iloc[0]
        if pd.isna(v):
            raise QueryError(f"'{raw}' is not a date I can read for column '{col}' (try YYYY-MM-DD)")
        d = _dates(s)
        if len(raw) <= 10:                                   # a whole day: '<=' means 'up to the end of that day'
            end = v + pd.Timedelta(days=1)
            return {"==": (d >= v) & (d < end), "!=": ((d < v) | (d >= end)) & d.notna(),
                    ">": d >= end, ">=": d >= v, "<": d < v, "<=": d < end}[f.op]
        return {"==": d == v, "!=": (d != v) & d.notna(), ">": d > v, ">=": d >= v, "<": d < v, "<=": d <= v}[f.op]

    if kind == "boolean":
        truth = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
        raw = _need_value(f).lower()
        if f.op not in ("==", "!=") or raw not in truth:
            raise QueryError(f"column '{col}' is true/false: use == or != with true or false")
        eq = s == truth[raw]
        return eq if f.op == "==" else (~eq) & s.notna()

    # text
    n = normalize_text(s)
    if f.op in (">", ">=", "<", "<="):
        raise QueryError(f"'{f.op}' needs numbers or dates, but '{col}' is a text column")
    if f.op == "in":
        wanted = [normalize_one(v) for v in (f.values or [_need_value(f)])]
        return n.isin(wanted).fillna(False)
    wanted_one = normalize_one(_need_value(f))
    if f.op == "contains":
        return n.str.contains(re.escape(wanted_one), regex=True).fillna(False)
    eq = (n == wanted_one).fillna(False)
    return eq if f.op == "==" else (~eq) & s.notna()


def _squash(s: pd.Series) -> pd.Series:
    """Like normalize_text, then punctuation and stray symbols removed too, so 'São Paulo' and the garbled
    'Sí£o Paulo' end up as 'sao paulo' and 'sio paulo' (one letter apart)."""
    return normalize_text(s).str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def _lookup(s: pd.Series) -> dict[str, str]:
    raw = s.dropna().astype("string").drop_duplicates()
    return dict(zip(_squash(raw).tolist(), raw.tolist()))          # squashed -> an original spelling


def _closest_values(s: pd.Series, wanted: list[str], n: int = 3) -> list[str]:
    """Real values of the column nearest to what was asked for (accent/case/symbol-insensitive)."""
    lookup = _lookup(s)
    out: list[str] = []
    for w in wanted:
        for m in difflib.get_close_matches(_squash(pd.Series([w])).iloc[0], list(lookup), n=n, cutoff=0.6):
            if lookup[m] not in out:
                out.append(lookup[m])
    return out[:n]


NEAR_MATCH_CUTOFF = 0.85       # how alike a value must be to be used in place of one that does not exist
NEAR_MATCH_GAP = 0.08          # ...and how much better than the runner-up, so the choice is not a coin flip


def _unique_near_match(s: pd.Series, wanted: str) -> str | None:
    """The one existing value that is clearly what `wanted` meant (typo, accent, garbled characters), else None."""
    lookup = _lookup(s)
    w = _squash(pd.Series([wanted])).iloc[0]
    close = difflib.get_close_matches(w, list(lookup), n=2, cutoff=NEAR_MATCH_CUTOFF)
    if not close:
        return None
    if len(close) == 2:
        sm = difflib.SequenceMatcher
        if sm(None, w, close[0]).ratio() - sm(None, w, close[1]).ratio() < NEAR_MATCH_GAP:
            return None
    return lookup[close[0]]


def _describe(req: TableRequest, column: str | None, filters: list[tuple[Filter, str]], groups: list[str]) -> str:
    what = _OP_WORDS[req.operation] + (f" '{column}'" if column else " rows")
    if req.operation == "count" and column:
        what = f"count of non-empty '{column}'"
    parts = [what]
    if filters:
        conds = []
        for f, c in filters:
            val = f.values if f.op == "in" else f.value
            conds.append(f"{c} {f.op}" + ("" if f.op in ("is_null", "not_null") else f" {_quote(val)}"))
        parts.append("where " + " and ".join(conds))
    if groups:
        parts.append("grouped by " + ", ".join(groups))
    return " ".join(parts)


# ------------------------------------------------------------------ running one request
def run_request(req: TableRequest, info: TableInfo, df: pd.DataFrame, ref: str) -> TableResult:
    base = dict(ref=ref, table_ref=info.ref, topic_id=info.topic_id, source=info.source, table_id=info.table_id,
                rows_total=len(df))
    desc = f"{req.operation} (request not understood)"
    try:
        cols = list(df.columns)
        column = resolve_column(req.column, cols) if req.column else None
        if req.operation != "count" and column is None:
            raise QueryError(f"'{req.operation}' needs a column")
        groups = [resolve_column(g, cols) for g in req.group_by[:MAX_GROUP_COLUMNS]]
        kinds: dict[str, str] = {}

        def kind(c):
            if c not in kinds:
                kinds[c] = column_kind(df[c])
            return kinds[c]

        resolved = [(f, resolve_column(f.column, cols)) for f in req.filters]
        desc = _describe(req, column, resolved, groups)

        notes: list[str] = []
        suggestions: dict[str, list[str]] = {}
        mask = pd.Series(True, index=df.index)
        applied: list[tuple[Filter, str]] = []
        for f, c in resolved:
            m = _filter_mask(df, f, c, kind(c))
            if not m.any() and kind(c) == "text" and f.op == "==" and f.value:
                fix = _unique_near_match(df[c], f.value)             # a clear typo / accent / garbled-text near miss
                if fix is not None:
                    notes.append(f"'{f.value}' does not exist in '{c}'; used the closest existing value '{fix}'")
                    f = f.model_copy(update={"value": fix})
                    m = _filter_mask(df, f, c, "text")
            if not m.any() and kind(c) == "text" and f.op in ("==", "in", "contains"):
                close = _closest_values(df[c], f.values if f.op == "in" and f.values else [f.value or ""])
                if close:
                    suggestions[c] = close
            applied.append((f, c))
            mask &= m
        matched = df[mask]
        desc = _describe(req, column, applied, groups)               # describes what was really computed

        common = dict(base, description=desc, ok=True, rows_matched=len(matched), notes=notes, suggestions=suggestions)

        # ---- the series to aggregate
        op = req.operation
        if column is None:
            vals = pd.Series(1, index=matched.index)                      # counting rows
        elif op in ("sum", "mean", "median"):
            vals = _numeric(matched[column], kind(column), column, notes)
        elif op in ("min", "max"):
            k = kind(column)
            vals = _dates(matched[column]) if k == "date" else _numeric(matched[column], k, column, notes)
        else:                                                              # count of non-empty / nunique: any column
            vals = matched[column]

        pandas_op = {"count": "count", "nunique": "nunique"}.get(op, op)
        if column is None:
            pandas_op = "sum"                                              # sum of ones = number of rows

        # ---- one figure
        if not groups:
            if matched.empty and op != "count":
                return TableResult(**common, value=None)
            if column is None:
                return TableResult(**common, value=len(matched))
            return TableResult(**common, value=_native(getattr(vals, pandas_op)()))

        # ---- grouped figures
        keys = [matched[g] for g in groups]
        res = getattr(vals.groupby(keys, dropna=True, sort=False), pandas_op)().rename("value").reset_index()
        res = res.dropna(subset=["value"])
        ascending = req.order == "asc"
        res = res.sort_values(["value", *groups], ascending=[ascending] + [True] * len(groups), kind="stable")
        limit = max(1, min(req.limit or DEFAULT_LIMIT, MAX_LIMIT))
        direction = "smallest first" if ascending else "largest first"
        rows_note = (f"{len(res):,} group(s) in total, sorted {direction}; "
                     + (f"only the first {limit} are shown, the others are cut off" if len(res) > limit else "all are shown"))
        rows = [{**{g: _native(r[g]) for g in groups}, "value": _native(r["value"])} for _, r in res.head(limit).iterrows()]
        return TableResult(**common, rows=rows, rows_note=rows_note)

    except QueryError as e:
        return TableResult(**base, description=desc, ok=False, error=str(e), suggestions=e.suggestions)


def _find_table(ref: str, catalog: Sequence[TableInfo]) -> TableInfo | None:
    r = ref.strip().lower()
    for t in catalog:
        if r in (t.ref.lower(), t.title.lower(), f"{t.topic_id}/{t.source}".lower(), t.source.lower()):
            return t
    return None


def execute_requests(
    requests: Sequence[TableRequest],
    catalog: Sequence[TableInfo],
    start: int = 1,
    loader: Callable[[str, str, str], pd.DataFrame] = assets.load_table,
) -> list[TableResult]:
    """Run up to MAX_REQUESTS requests. Never raises: a problem becomes a failed TableResult the model can read."""
    results: list[TableResult] = []
    cache: dict[str, pd.DataFrame] = {}
    for i, req in enumerate(list(requests)[:MAX_REQUESTS]):
        ref = f"C{start + i}"
        info = _find_table(req.table, catalog)
        if info is None:
            valid = ", ".join(t.ref for t in catalog) or "none"
            results.append(TableResult(ref, req.table, "", "", "", f"{req.operation} on '{req.table}'", False,
                                       error=f"there is no table '{req.table}' (available: {valid})"))
            continue
        try:
            if info.ref not in cache:
                cache[info.ref] = loader(info.topic_id, info.source, info.table_id)
            results.append(run_request(req, info, cache[info.ref], ref))
        except Exception as e:                                           # a broken file etc.: report, don't crash the answer
            log.exception("table_query failed for %s", info.title)
            results.append(TableResult(ref, info.ref, info.topic_id, info.source, info.table_id,
                                       f"{req.operation} on {info.title}", False, error=f"{type(e).__name__}: {e}"))
    return results
