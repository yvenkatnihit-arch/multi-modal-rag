"""The project's shape, enforced.

    src/core        shared by both pipelines          may import: core
    src/ingestion   data -> searchable store          may import: core, ingestion
    src/query       question -> answer                may import: core, query
    src/config.py   settings (imported by everything)

So the two pipelines never depend on each other: they meet only through what ingestion writes (the vector store and
assets/) and what the query side reads. Breaking that makes either pipeline impossible to run, test or replace alone.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGES = {"core": {"core"}, "ingestion": {"core", "ingestion"}, "query": {"core", "query"}}


def imported_packages(path: Path) -> set[str]:
    """Which of our packages (core / ingestion / query / config) does this file import?"""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + ([f"{node.module}.{a.name}" for a in node.names] if node.module == "src" else [])
        for name in names:
            parts = name.split(".")
            if parts[0] == "src" and len(parts) > 1:
                found.add(parts[1])
    return found


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_each_package_imports_only_what_it_is_allowed_to(package):
    allowed = PACKAGES[package] | {"config"}
    offenders = []
    for path in sorted((SRC / package).rglob("*.py")):
        extra = imported_packages(path) - allowed
        if extra:
            offenders.append(f"{path.relative_to(ROOT).as_posix()} imports {sorted(extra)}")
    assert not offenders, f"{package} may only import {sorted(PACKAGES[package])}:\n  " + "\n  ".join(offenders)


def test_every_module_lives_in_exactly_one_package_and_nothing_is_left_loose():
    loose = sorted(p.name for p in SRC.glob("*.py") if p.name not in ("__init__.py", "config.py"))
    assert not loose, f"modules directly under src/ (put them in core, ingestion or query): {loose}"
    for package in PACKAGES:
        assert (SRC / package / "__init__.py").exists(), f"src/{package} is not a package"
    assert {p.name for p in SRC.iterdir() if p.is_dir() and p.name != "__pycache__"} == set(PACKAGES)


def test_the_entry_points_sit_at_the_project_root():
    for name in ("app.py", "ask.py", "ingest.py", "check_setup.py", "requirements.txt", "README.md"):
        assert (ROOT / name).exists(), name


def test_the_tests_mirror_the_source_layout():
    folders = {p.name for p in (ROOT / "tests").iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert set(PACKAGES) <= folders, "each src package should have a matching tests folder"
    stray = sorted(p.name for p in (ROOT / "tests").glob("test_*.py") if p.name != "test_architecture.py")
    assert not stray, f"tests directly under tests/ (put them in the folder of the code they test): {stray}"
