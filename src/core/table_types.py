"""Column-type helpers shared by the table parser (describing a column) and table_query (computing on it).

Keeping one definition means '$6.60' is a number, and '2015-02-24 11:35:52 -0800' a date, in both places.
"""
import re

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

_DATE_HINT = r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
_AMOUNT = re.compile(r"^\s*(?:[A-Za-z]{0,3}\$|[€£¥₹]|RM|Rs\.?)?\s*-?\d[\d,]*(?:\.\d+)?\s*%?\s*$")


def to_number(s: pd.Series) -> pd.Series:
    """'$1,200.50' -> 1200.5. Anything that is not a number becomes NaN."""
    cleaned = s.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def is_amount_text(nn: pd.Series) -> bool:
    """Numbers stored as text, e.g. '$6.60' or '1,200'. Pure digit strings (zip codes, ids) do not count."""
    sample = nn.astype(str).head(200)
    if len(sample) < 3 or sample.str.fullmatch(r"\d+").all():
        return False
    return sample.str.match(_AMOUNT).mean() >= 0.95


def is_date_like(nn: pd.Series) -> bool:
    sample = nn.astype(str).head(50)
    if len(sample) < 3 or sample.str.contains(_DATE_HINT).mean() < 0.9:
        return False
    return pd.to_datetime(sample, errors="coerce", utc=True, format="mixed").notna().mean() >= 0.9


def column_kind(s: pd.Series) -> str:
    """empty | boolean | integer | decimal | amount_text | date | text"""
    nn = s.dropna()
    if nn.empty:
        return "empty"
    if is_bool_dtype(s):
        return "boolean"
    if is_numeric_dtype(s):
        return "integer" if is_integer_dtype(s) or (nn % 1 == 0).all() else "decimal"
    if is_amount_text(nn):
        return "amount_text"
    if is_date_like(nn):
        return "date"
    return "text"


def normalize_text(s: pd.Series) -> pd.Series:
    """For tolerant comparison: accents folded, case-insensitive, spaces collapsed ('São  Paulo' == 'sao paulo')."""
    t = s.astype("string").str.normalize("NFKD").str.replace(r"[̀-ͯ]", "", regex=True)
    return t.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_one(value: str) -> str:
    return normalize_text(pd.Series([value])).iloc[0]
