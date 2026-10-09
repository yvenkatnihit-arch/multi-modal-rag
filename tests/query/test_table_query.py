import pandas as pd
import pytest

from src import config
from src.core import assets
from src.query.table_query import (MAX_REQUESTS, Filter, TableRequest, build_catalog, catalog_text, execute_requests,
                             resolve_column, QueryError)

SALES = pd.DataFrame({
    "city": ["New Delhi", "New Delhi", "Mumbai", "Mumbai", "Mumbai", "Sí£o Paulo"],     # note the garbled name, as in the real data
    "rating": [4.0, 3.0, 5.0, 4.5, None, 2.0],
    "votes": [10, 20, 30, 40, 50, 60],
    "delivery": ["Yes", "No", "Yes", "Yes", "No", "Yes"],
    "when": ["2015-02-20 10:00:00 -0800", "2015-02-21 09:00:00 -0800", "2015-02-22 08:00:00 -0800",
             "2015-02-23 07:00:00 -0800", "2015-02-24 06:00:00 -0800", "2015-02-25 05:00:00 -0800"],
    "open": [True, False, True, True, False, True],
})
PRICES = pd.DataFrame({"item": [f"i{n}" for n in range(30)],
                       "price": [f"${n}.50" for n in range(29)] + ["n/a"]})            # 29 amounts + 1 junk value


@pytest.fixture(autouse=True)
def isolated_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")
    assets.save_table(SALES, "t", "sales.csv", "main")
    assets.save_table(PRICES, "t", "prices.xlsx", "Sheet1")


CATALOG = lambda: build_catalog(["t"])                  # noqa: E731


def q(operation, table="T2", **kw):
    """T1 = prices.xlsx, T2 = sales.csv (sorted by source name)."""
    return TableRequest(table=table, operation=operation, **kw)


def run(*requests, start=1):
    return execute_requests(list(requests), CATALOG(), start)


def one(request):
    return run(request)[0]


def where(column, op, value=None, values=()):
    return Filter(column=column, op=op, value=value, values=list(values))


# ------------------------------------------------------------------ the catalog
def test_catalog_lists_saved_tables_with_refs_row_counts_and_column_kinds():
    cat = CATALOG()
    assert [(t.ref, t.source, t.table_id, t.rows) for t in cat] == [("T1", "prices.xlsx", "Sheet1", 30), ("T2", "sales.csv", "main", 6)]
    assert cat[1].columns["rating"] == "decimal" and cat[1].columns["votes"] == "integer"
    assert cat[1].columns["when"] == "date" and cat[1].columns["city"] == "text" and cat[0].columns["price"] == "amount_text"
    text = catalog_text(cat)
    assert "T1: t/prices.xlsx [sheet Sheet1], 30 rows." in text and "price (number stored as text)" in text
    assert "T2: t/sales.csv, 6 rows." in text


def test_manifest_prune_removes_stale_tables_and_orphans():
    assert assets.prune_tables("t", {("sales.csv", "main")}) == [assets.table_asset_path("t", "prices.xlsx", "Sheet1").name]
    assert [e["source"] for e in assets.list_tables("t")] == ["sales.csv"]
    (assets.table_asset_path("t", "x.csv", "main")).write_bytes(b"orphan")                  # a parquet the manifest does not know
    assert len(assets.prune_tables("t", {("sales.csv", "main")})) == 1


def test_a_vanished_parquet_disappears_from_the_catalog():
    assets.table_asset_path("t", "sales.csv", "main").unlink()
    assert [t.source for t in CATALOG()] == ["prices.xlsx"]


# ------------------------------------------------------------------ aggregates
def test_counts():
    assert one(q("count")).value == 6
    r = one(q("count", filters=[where("city", "==", "new delhi")]))                      # case-insensitive
    assert r.value == 2 and r.rows_matched == 2 and r.rows_total == 6 and r.ok
    assert one(q("count", column="rating")).value == 5                                   # non-empty values only
    assert one(q("nunique", column="city")).value == 3


def test_numeric_aggregates_ignore_missing_values():
    f = [where("city", "==", "Mumbai")]
    assert one(q("mean", column="rating", filters=f)).value == 4.75                       # (5 + 4.5) / 2, the None is skipped
    assert one(q("sum", column="votes", filters=[where("votes", ">=", "30")])).value == 180
    assert one(q("median", column="votes")).value == 35
    assert (one(q("min", column="rating")).value, one(q("max", column="rating")).value) == (2, 5)


def test_text_filters_in_contains_not_equal_and_nulls():
    assert one(q("count", filters=[where("city", "in", values=["mumbai", "NEW DELHI"])])).value == 5
    assert one(q("count", filters=[where("city", "contains", "del")])).value == 2
    assert one(q("count", filters=[where("city", "!=", "Mumbai")])).value == 3
    assert one(q("count", filters=[where("rating", "is_null")])).value == 1
    assert one(q("count", filters=[where("rating", "not_null")])).value == 5


def test_numbers_stored_as_text_are_summed_and_the_junk_is_reported_not_hidden():
    r = one(q("sum", table="T1", column="price"))
    assert r.value == sum(n + 0.5 for n in range(29)) and any("1 value(s)" in n and "not numeric" in n for n in r.notes)


def test_boolean_and_date_filters():
    assert one(q("count", filters=[where("open", "==", "yes")])).value == 4
    assert one(q("count", filters=[where("when", ">=", "2015-02-23")])).value == 3
    assert one(q("count", filters=[where("when", "<=", "2015-02-21")])).value == 2         # '<= a day' includes that whole day
    assert one(q("count", filters=[where("when", "==", "2015-02-22")])).value == 1
    assert one(q("min", column="when")).value.startswith("2015-02-20T18:00:00")            # stored with -0800, shown in UTC


# ------------------------------------------------------------------ grouped results
def test_grouped_counts_are_sorted_and_limited():
    r = one(q("count", group_by=["city"]))
    assert r.rows == [{"city": "Mumbai", "value": 3}, {"city": "New Delhi", "value": 2}, {"city": "Sí£o Paulo", "value": 1}]
    assert one(q("count", group_by=["city"], limit=1)).rows == [{"city": "Mumbai", "value": 3}]
    assert one(q("count", group_by=["city"], order="asc")).rows[0] == {"city": "Sí£o Paulo", "value": 1}
    top = one(q("count", group_by=["city"], limit=1))
    assert top.rows_note == "3 group(s) in total, sorted largest first; only the first 1 are shown, the others are cut off"
    assert "groups (3 group(s) in total, sorted largest first; only the first 1" in top.to_text()
    assert one(q("count", group_by=["city"])).rows_note.endswith("all are shown")
    assert "smallest first" in one(q("count", group_by=["city"], order="asc")).rows_note


def test_grouping_by_two_columns_and_averaging_within_groups():
    r = one(q("sum", column="votes", group_by=["city", "delivery"]))
    assert {(x["city"], x["delivery"]): x["value"] for x in r.rows}[("Mumbai", "Yes")] == 70
    m = one(q("mean", column="rating", group_by=["city"]))
    assert {x["city"]: x["value"] for x in m.rows} == {"Mumbai": 4.75, "New Delhi": 3.5, "Sí£o Paulo": 2}


# ------------------------------------------------------------------ honesty when nothing matches or the request is wrong
def test_no_matching_rows_is_reported_not_turned_into_zero_or_a_made_up_average():
    f = [where("city", "==", "Paris")]
    mean = one(q("mean", column="rating", filters=f))
    assert mean.ok and mean.value is None and mean.rows_matched == 0
    assert "no rows matched" in mean.to_text()
    assert one(q("count", filters=f)).value == 0                                          # a count of nothing really is 0
    assert one(q("count", group_by=["city"], filters=f)).rows == []


def test_a_clear_near_miss_is_resolved_automatically_and_said_so():
    r = one(q("count", filters=[where("city", "==", "São Paulo")]))                        # the data holds 'Sí£o Paulo'
    assert r.value == 1 and r.rows_matched == 1
    assert r.notes == ["'São Paulo' does not exist in 'city'; used the closest existing value 'Sí£o Paulo'"]
    assert r.description == "count of rows where city == 'Sí£o Paulo'"                     # describes what was REALLY computed
    assert "note: 'São Paulo' does not exist" in r.to_text()
    assert one(q("count", filters=[where("city", "==", "Nw Delhi")])).value == 2           # a typo
    assert one(q("count", filters=[where("city", "==", "Sí£o Paulo")])).notes == []        # an exact match needs no note


def test_a_vague_near_miss_is_not_guessed_but_suggestions_are_offered():
    r = one(q("count", filters=[where("city", "==", "Sao Pablo")]))                        # alike, but not clearly the same
    assert r.value == 0 and r.notes == [] and r.suggestions == {"city": ["Sí£o Paulo"]}
    assert "closest existing values in 'city': 'Sí£o Paulo'" in r.to_text()
    assert one(q("count", filters=[where("city", "==", "Delhi")])).value == 0              # 'New Delhi' is too different
    assert one(q("count", filters=[where("city", "in", values=["São Paulo"])])).value == 0  # 'in' is never auto-corrected


def test_two_equally_close_values_are_never_auto_picked(tmp_path):
    two = pd.DataFrame({"name": ["Cafe Alpha", "Cafe Alpho", "Other"], "n": [1, 2, 3]})
    assets.save_table(two, "t", "two.csv", "main")
    cat = build_catalog(["t"])
    ref = next(t.ref for t in cat if t.source == "two.csv")
    [r] = execute_requests([TableRequest(table=ref, operation="count", filters=[where("name", "==", "Cafe Alphe")])], cat)
    assert r.value == 0 and r.notes == [] and set(r.suggestions["name"]) == {"Cafe Alpha", "Cafe Alpho"}


def test_unknown_column_fails_with_the_closest_names():
    r = one(q("mean", column="ratng"))
    assert not r.ok and "no column named 'ratng'" in r.error and "rating" in r.suggestions["columns"]
    assert "ERROR" in r.to_text()


def test_column_names_resolve_case_and_punctuation_insensitively():
    assert resolve_column("RATING", ["rating", "votes"]) == "rating"
    assert resolve_column("Has Online Delivery", ["Has Online delivery"]) == "Has Online delivery"
    with pytest.raises(QueryError):
        resolve_column("nothing", ["rating"])


@pytest.mark.parametrize("request_,fragment", [
    (q("sum", column="city"), "holds text values"),
    (q("mean", column="when"), "holds date values"),
    (q("min", column="city"), "holds text values"),
    (q("sum"), "needs a column"),
    (q("count", filters=[where("city", ">", "M")]), "needs numbers or dates"),
    (q("count", filters=[where("votes", "contains", "1")]), "only works on text"),
    (q("count", filters=[where("votes", "==", "lots")]), "is not a number"),
    (q("count", filters=[where("when", ">", "never")]), "not a date"),
    (q("count", filters=[where("city", "==")]), "needs a value"),
    (q("count", filters=[where("open", ">", "true")]), "true/false"),
])
def test_invalid_requests_come_back_as_readable_errors_never_exceptions(request_, fragment):
    r = one(request_)
    assert not r.ok and fragment in r.error


def test_values_that_look_like_code_are_just_text():
    r = one(q("count", filters=[where("city", "==", "__import__('os').system('echo hacked')")]))
    assert r.ok and r.value == 0


# ------------------------------------------------------------------ plumbing
def test_tables_can_be_named_by_ref_file_or_topic_path_and_unknown_ones_list_the_valid_refs():
    for name in ("T2", "t2", "sales.csv", "t/sales.csv"):
        assert one(q("count", table=name)).value == 6
    r = one(q("count", table="T9"))
    assert not r.ok and "no table 'T9'" in r.error and "T1, T2" in r.error


def test_at_most_three_requests_run_and_refs_continue_from_start():
    results = run(*[q("count")] * 5, start=4)
    assert len(results) == MAX_REQUESTS == 3 and [r.ref for r in results] == ["C4", "C5", "C6"]
    assert [r.number for r in results] == [4, 5, 6]


def test_a_broken_table_file_becomes_a_failed_result():
    def boom(*a):
        raise OSError("disk unreadable")
    [r] = execute_requests([q("count")], CATALOG(), loader=boom)
    assert not r.ok and "disk unreadable" in r.error


def test_result_text_and_dict_describe_what_was_computed():
    r = one(q("mean", column="rating", filters=[where("city", "==", "Mumbai")]))
    assert r.description == "average (mean) of 'rating' where city == 'Mumbai'"
    text = r.to_text()
    assert "C1 [T2: t/sales.csv] average (mean) of 'rating' where city == 'Mumbai'" in text and "result: 4.75" in text
    assert "computed over 3 of 6 rows of the full table" in text
    g = one(q("count", group_by=["city"])).to_text()
    assert "city = Mumbai -> 3" in g
    d = r.to_dict()
    assert d["value"] == 4.75 and d["rows_matched"] == 3 and d["ok"] is True
