import pymupdf
import pytest

from src import config
from src.core import assets
from src.ingestion.chunker import chunk_records
from src.query.citations import _location
from src.ingestion.file_router import RoutedFile
from src.ingestion.parsers.pdf_parser import parse_pdf
from src.core.records import Metadata, make_label
from src.query.table_query import TableRequest, build_catalog, execute_requests
from tests.helpers import html_pdf

CSS = "<style>table{border-collapse:collapse;} th,td{border:1px solid #000000;padding:5px;}</style>"
SURVEY = [("Nairobi", 34, 5), ("Mombasa", 55, 12), ("Kisumu", 48, 9), ("Nakuru", 33, 26), ("Eldoret", 89, 7)]


@pytest.fixture(autouse=True)
def isolated_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(config, "ROOT", tmp_path)


def table(rows, header=("City", "Sightings", "Relocations")):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def pdf(tmp_path, *page_bodies, name="survey.pdf"):
    """One PDF page per body (HTML), with ruled-table styling."""
    return RoutedFile(html_pdf(tmp_path, page_bodies, name, CSS), name, "pdf")


INTRO = "<h1>Urban Wildlife Survey</h1><p>Snake sightings reported by residents during 2025 in five cities.</p><h2>Sightings and relocations</h2>"


def by_modality(records, modality):
    return [r for r in records if r.metadata.modality == modality]


def test_a_ruled_table_becomes_table_records_with_its_page_and_a_pdf_table_id(tmp_path):
    recs = parse_pdf(pdf(tmp_path, INTRO + table(SURVEY)), "wildlife", ocr=False)
    tbl = by_modality(recs, "table")
    assert {r.metadata.table_id for r in tbl} == {"p1t1"} and {r.metadata.page for r in tbl} == {1}
    assert [r.metadata.part for r in tbl][0] == "summary" and any(r.metadata.part.startswith("rows-") for r in tbl)
    summary = tbl[0].text
    assert "Table survey.pdf (page 1, table 1) (topic wildlife): 5 rows x 3 columns." in summary
    assert "Sightings (integer): min 33, max 89" in summary
    assert make_label(tbl[0].metadata) == "table survey.pdf p.1 t1 (summary)"
    assert _location(tbl[0].metadata) == "p.1, table 1, summary"


def test_the_full_table_is_saved_with_real_numbers_so_table_query_can_calculate_on_it(tmp_path):
    parse_pdf(pdf(tmp_path, INTRO + table(SURVEY)), "wildlife", ocr=False)
    saved = assets.load_table("wildlife", "survey.pdf", "p1t1")
    assert list(saved.columns) == ["City", "Sightings", "Relocations"]
    assert saved["Sightings"].dtype.kind == "i" and int(saved["Sightings"].sum()) == 259
    catalog = build_catalog(["wildlife"])
    assert catalog[0].title == "wildlife/survey.pdf [page 1, table 1]"
    assert catalog[0].columns == {"City": "text", "Sightings": "integer", "Relocations": "integer"}
    [total, top] = execute_requests([
        TableRequest(table="T1", operation="sum", column="Sightings"),
        TableRequest(table="T1", operation="max", column="Relocations", group_by=["City"], limit=1)], catalog)
    assert total.value == 259 and top.rows == [{"City": "Nakuru", "value": 26}]


def test_the_page_text_no_longer_repeats_the_table_but_keeps_everything_around_it(tmp_path):
    f = pdf(tmp_path, INTRO + table(SURVEY))
    with_tables = by_modality(parse_pdf(f, "t", ocr=False), "text")
    text = " ".join(r.text for r in with_tables)
    assert "Urban Wildlife Survey" in text and "Sightings and relocations" in text
    assert "Mombasa" not in text and "Eldoret" not in text                                  # the table's own words moved out
    before = " ".join(r.text for r in parse_pdf(f, "t", ocr=False, tables=False))
    assert "Mombasa" in before and not by_modality(parse_pdf(f, "t", ocr=False, tables=False), "table")


def test_two_tables_on_one_page_and_tables_on_later_pages_get_their_own_ids_and_pages(tmp_path):
    page1 = INTRO + table(SURVEY) + "<p>Second table follows.</p>" + table([("Alpha", 1), ("Beta", 2)], ("Team", "Score"))
    page2 = "<h2>Appendix</h2><p>Staffing numbers by region.</p>" + table([("North", 10, 3), ("South", 20, 4)], ("Region", "Staff", "Vehicles"))
    recs = parse_pdf(pdf(tmp_path, page1, page2), "t", ocr=False)
    ids = {(r.metadata.page, r.metadata.table_id) for r in by_modality(recs, "table")}
    assert ids == {(1, "p1t1"), (1, "p1t2"), (2, "p2t1")}
    assert [r.metadata.page for r in recs] == sorted(r.metadata.page for r in recs)         # ordered by page
    assert [(r.metadata.modality, r.metadata.page) for r in recs][:2] == [("text", 1), ("table", 1)]   # a page's text first
    assert len(assets.list_tables("t")) == 3
    assert int(assets.load_table("t", "survey.pdf", "p2t1")["Staff"].sum()) == 30


def test_numbers_become_numeric_only_when_every_cell_is_clean_and_codes_stay_text(tmp_path):
    rows = [("007", "1,200", "ok", "10"), ("010", "35.5", "n/a", "20"), ("123", "8", "ok", "30")]
    parse_pdf(pdf(tmp_path, table(rows, ("Code", "Amount", "Flag", "Qty"))), "t", ocr=False)
    saved = assets.load_table("t", "survey.pdf", "p1t1")
    assert saved["Code"].tolist() == ["007", "010", "123"]                                  # leading zeros: an identifier
    assert saved["Amount"].tolist() == [1200.0, 35.5, 8.0]                                  # thousands comma handled
    assert saved["Flag"].tolist() == ["ok", "n/a", "ok"] and saved["Qty"].tolist() == [10, 20, 30]


def test_layout_boxes_and_mostly_empty_tables_are_not_treated_as_data(tmp_path):
    mostly_empty = table([("", "", ""), ("x", "", ""), ("", "", "")], ("a", "b", "c"))
    recs = parse_pdf(pdf(tmp_path, "<p>Some prose on the page, long enough to be kept as text.</p>" + mostly_empty),
                     "t", ocr=False)
    assert not by_modality(recs, "table") and by_modality(recs, "text")


def test_repeated_header_names_do_not_break_saving(tmp_path):
    parse_pdf(pdf(tmp_path, table([("a", 1), ("b", 2)], ("Name", "Name"))), "t", ocr=False)
    assert list(assets.load_table("t", "survey.pdf", "p1t1").columns) == ["Name", "Name_2"]


def test_chunk_ids_of_text_and_table_records_of_one_pdf_never_collide(tmp_path):
    recs = parse_pdf(pdf(tmp_path, INTRO + table(SURVEY)), "t", ocr=False)
    ids = [c.id for c in chunk_records(recs)]
    assert len(ids) == len(set(ids)) and any("/p1-tp1t1-summary/" in i for i in ids) and any(i.endswith("/p1/0") for i in ids)


def test_a_pdf_without_ruled_tables_is_unaffected(tmp_path):
    recs = parse_pdf(pdf(tmp_path, "<h1>Notes</h1><p>Just a paragraph of ordinary prose with no table at all in it.</p>"), "t", ocr=False)
    assert [r.metadata.modality for r in recs] == ["text"]
    assert Metadata("t", "x.pdf", "table", table_id="main").table_id == "main"            # other tables keep their naming


def test_the_generated_survey_pdf_in_the_data_folder_has_its_table_extracted():
    from pathlib import Path
    p = config.DATA_DIR / "wildlife_reports" / "urban_wildlife_survey_2025.pdf"
    if not p.exists():
        pytest.skip("sample data not present")
    recs = parse_pdf(RoutedFile(p, p.name, "pdf"), "wildlife_reports", ocr=False, images=False)
    saved = assets.load_table("wildlife_reports", p.name, "p1t1")
    assert int(saved.iloc[:, 1].sum()) == 259 and int(saved.iloc[:, 2].sum()) == 59       # the facts recorded at generation time
    assert "Mombasa" not in " ".join(r.text for r in by_modality(recs, "text"))
