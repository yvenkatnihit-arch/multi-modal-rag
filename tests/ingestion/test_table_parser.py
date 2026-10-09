import json

import pandas as pd
import pytest

from src import config
from src.core.assets import load_table
from src.ingestion.chunker import chunk_records
from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.table_parser import parse_table


@pytest.fixture(autouse=True)
def isolated_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")


def parse(tmp_path, name, content: bytes | str | None = None, df: pd.DataFrame | None = None, **kw):
    p = tmp_path / name
    if df is not None:
        if name.endswith(".xlsx"):
            df.to_excel(p, index=False)
        else:
            df.to_csv(p, index=False, **kw)
    else:
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return parse_table(RoutedFile(p, name, "table"), "t")


def by_part(records):
    return {r.metadata.part: r for r in records}


SALES = pd.DataFrame({
    "region": ["North", "South", "North", "East"] * 5,
    "units": list(range(1, 21)),
    "price": [2.5, 3.0, 4.5, 1.25] * 5,
})


def test_small_csv_gives_summary_plus_every_row_with_header_repeated(tmp_path):
    recs = parse(tmp_path, "sales.csv", df=SALES)
    parts = by_part(recs)
    assert "summary" in parts and parts["summary"].metadata.table_id == "main"
    row_recs = [r for r in recs if r.metadata.part.startswith("rows-")]
    assert all(r.text.split("\n")[1] == "row | region | units | price" for r in row_recs)
    embedded = [int(l.split(" | ")[0]) for r in row_recs for l in r.text.split("\n")[2:]]
    assert embedded == list(range(1, 21))
    assert all(r.metadata.modality == "table" for r in recs)


def test_summary_has_shape_and_column_statistics(tmp_path):
    s = by_part(parse(tmp_path, "sales.csv", df=SALES))["summary"].text
    assert "20 rows x 3 columns" in s
    assert "units (integer): min 1, max 20, mean 10.5, median 10.5" in s
    assert "price (decimal): min 1.25, max 4.5" in s
    assert "region (text): values: North (10), South (5), East (5)" in s


def test_full_table_is_stored_in_assets_with_types(tmp_path):
    parse(tmp_path, "sales.csv", df=SALES)
    stored = load_table("t", "sales.csv", "main")
    assert len(stored) == 20 and stored["units"].sum() == 210 and stored["price"].dtype.kind == "f"


def test_big_table_is_sampled_but_stored_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLE_EMBED_ROWS", 50)
    big = pd.DataFrame({"id": range(1, 1001), "v": [i % 7 for i in range(1000)]})
    recs = parse(tmp_path, "big.csv", df=big)
    sample_recs = [r for r in recs if r.metadata.part.startswith("sample-")]
    assert sample_recs and not any(r.metadata.part.startswith("rows-") for r in recs)   # honest naming
    embedded = [int(l.split(" | ")[0]) for r in sample_recs for l in r.text.split("\n")[2:]]
    assert len(embedded) == 50 and embedded == sorted(embedded)
    assert sample_recs[0].text.startswith("Table big.csv, sampled rows ")
    assert "only a sample of 50 of 1,000 rows" in by_part(recs)["summary"].text
    assert len(load_table("t", "big.csv", "main")) == 1000
    assert [r.text for r in parse(tmp_path, "big.csv", df=big)] == [r.text for r in recs]   # deterministic sample


def test_pipes_newlines_and_long_cells_are_tamed_in_row_text(tmp_path):
    df = pd.DataFrame({"note": ["a|b\nc", "x" * 500], "n": [1, 2]})
    recs = parse(tmp_path, "n.csv", df=df)
    rows = [l for r in recs if r.metadata.part.startswith("rows-") for l in r.text.split("\n")[2:]]
    assert rows[0] == "1 | a/b c | 1"
    assert len(rows[1]) < 200 and rows[1].count(" | ") == 2
    assert load_table("t", "n.csv", "main")["note"][1] == "x" * 500       # the stored copy is untouched


def test_semicolon_delimiter_and_latin1_encoding(tmp_path):
    recs = parse(tmp_path, "eu.csv", "name;city\nJosé;Zürich\nAnna;Köln\n".encode("latin-1"))
    assert "2 rows x 2 columns" in by_part(recs)["summary"].text
    assert "José" in " ".join(r.text for r in recs)


def test_excel_one_table_per_sheet_and_empty_sheet_skipped(tmp_path):
    p = tmp_path / "book.xlsx"
    with pd.ExcelWriter(p) as w:
        SALES.to_excel(w, sheet_name="Sales 2024", index=False)
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="Small", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Empty", index=False)
    recs = parse_table(RoutedFile(p, "book.xlsx", "table"), "t")
    assert {r.metadata.table_id for r in recs} == {"Sales_2024", "Small"}
    assert len(load_table("t", "book.xlsx", "Sales_2024")) == 20
    assert "[sheet Sales_2024]" in by_part([r for r in recs if r.metadata.table_id == "Sales_2024"])["summary"].text


def test_json_records_are_flattened_and_wrapper_key_stripped(tmp_path):
    data = {"results_shown": 2, "restaurants": [
        {"restaurant": {"name": "Alpha", "user_rating": {"aggregate_rating": "4.3"}, "tags": ["x", "y"]}},
        {"restaurant": {"name": "Beta", "user_rating": {"aggregate_rating": "3.1"}, "tags": []}},
    ]}
    recs = parse(tmp_path, "r.json", json.dumps(data))
    stored = load_table("t", "r.json", "main")
    assert list(stored.columns) == ["name", "user_rating.aggregate_rating", "tags"]
    assert stored["tags"][0] == '["x", "y"]'
    assert "Alpha" in " ".join(r.text for r in recs)


def test_unnamed_index_column_dropped_and_empty_table_skipped(tmp_path):
    csv = ",a,b\n0,1,2\n1,3,4\n"
    parse(tmp_path, "i.csv", csv)
    assert list(load_table("t", "i.csv", "main").columns) == ["a", "b"]
    assert parse(tmp_path, "h.csv", "a,b\n") == []


def test_date_and_missing_values_described(tmp_path):
    df = pd.DataFrame({"when": ["2015-02-24 11:35:52 -0800", "2015-02-22 12:01:01 -0800", "2015-02-23 01:00:00 -0800"],
                       "x": [1.0, None, 3.0]})
    s = by_part(parse(tmp_path, "d.csv", df=df))["summary"].text
    assert "when (date): from 2015-02-22 to 2015-02-24" in s
    assert "x (integer): min 1, max 3" in s and "1 missing" in s


def test_numbers_stored_as_text_are_recognised_but_ids_are_not(tmp_path):
    from src.ingestion.parsers.table_parser import to_number
    df = pd.DataFrame({"total": ["44.73", "$6.60", "1,200.50", "8.20"], "zip": ["02134", "10001", "94105", "60601"],
                       "name": ["a", "b", "c", "d"]})
    s = by_part(parse(tmp_path, "m.csv", df=df, quoting=1))["summary"].text
    assert "total (number stored as text" in s and "min 6.6, max 1200.5" in s
    assert "zip (number stored as text" not in s and "name (number stored" not in s
    assert load_table("t", "m.csv", "main")["total"][1] == "$6.60"          # the stored copy keeps the original
    assert to_number(pd.Series(["$6.60", "1,200.50", "x", None])).tolist()[:2] == [6.6, 1200.5]


def test_chunker_accepts_parsed_tables_and_ids_are_unique(tmp_path):
    chunks = chunk_records(parse(tmp_path, "sales.csv", df=SALES))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)) and ids[0] == "t/sales.csv/tmain-summary/0"
    assert any(i.startswith("t/sales.csv/tmain-rows-1-") for i in ids)
    assert chunks[0].label == "table sales.csv (summary)" and chunks[1].label.startswith("table sales.csv rows 1-")
