"""Safety net for the whole test suite: no test may reach the real Gemini API by accident.

Every test starts with the shared Gemini client replaced by an object that raises on any use. A test that really needs
a client (the client's own tests) builds its own. If you see this error, pass a fake analyzer / generator instead.
"""
import pytest


class _NoRealGemini:
    def __getattr__(self, name):
        raise AssertionError(f"a test tried to call the real Gemini API (client.{name}); inject a fake instead")


@pytest.fixture(autouse=True)
def forbid_real_gemini(monkeypatch):
    monkeypatch.setattr("src.core.gemini_client._client", _NoRealGemini())


@pytest.fixture(autouse=True)
def keep_real_project_data_untouched(tmp_path_factory, monkeypatch):
    """Anything a test writes to assets/, chroma_db/ or logs/ goes to a temp folder, never to the real project."""
    from src import config

    sandbox = tmp_path_factory.mktemp("sandbox")
    monkeypatch.setattr(config, "ASSETS_DIR", sandbox / "assets")
    monkeypatch.setattr(config, "CHROMA_DIR", sandbox / "chroma_db")
    monkeypatch.setattr(config, "LOGS_DIR", sandbox / "logs")
