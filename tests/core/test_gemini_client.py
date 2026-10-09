import httpx
import pytest
from google.genai import errors

from src.core import gemini_client
from src.core.gemini_client import MAX_RETRIES, call_with_retries, is_transient


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    slept = []
    monkeypatch.setattr(gemini_client.time, "sleep", slept.append)
    return slept


def flaky(*failures, result="ok"):
    """A function that raises each failure in turn, then returns `result`."""
    queue = list(failures)
    calls = []

    def fn():
        calls.append(1)
        if queue:
            raise queue.pop(0)
        return result

    fn.calls = calls
    return fn


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


@pytest.mark.parametrize("error", [httpx.ReadError("connection reset"), httpx.ConnectTimeout("slow"),
                                   ConnectionResetError("forcibly closed"), TimeoutError("t"), api_error(429), api_error(503)])
def test_transient_failures_are_retried_until_they_clear(error, no_waiting):
    fn = flaky(error, error)
    assert call_with_retries(fn) == "ok" and len(fn.calls) == 3
    assert no_waiting == [2.0, 4.0]                                  # exponential backoff


@pytest.mark.parametrize("error", [ValueError("bug"), api_error(400), api_error(403), KeyError("k")])
def test_other_failures_are_raised_immediately_without_retrying(error, no_waiting):
    fn = flaky(error)
    with pytest.raises(type(error)):
        call_with_retries(fn)
    assert len(fn.calls) == 1 and no_waiting == []


def test_gives_up_after_the_maximum_number_of_attempts(no_waiting):
    fn = flaky(*[httpx.ReadError("down")] * 20)
    with pytest.raises(httpx.ReadError):
        call_with_retries(fn)
    assert len(fn.calls) == MAX_RETRIES and len(no_waiting) == MAX_RETRIES - 1


def test_backoff_never_exceeds_a_minute(no_waiting):
    with pytest.raises(httpx.ReadError):
        call_with_retries(flaky(*[httpx.ReadError("down")] * 20))
    assert max(no_waiting) <= 60


def test_the_shared_client_is_created_once_even_when_many_threads_ask_at_the_same_moment(monkeypatch):
    import threading
    import time

    import google.genai

    built = []

    class SlowClient:
        def __init__(self, api_key):
            built.append(self)
            time.sleep(0.05)                                      # widens the window in which a race would show

    monkeypatch.setattr(google.genai, "Client", SlowClient)
    monkeypatch.setattr(gemini_client, "_client", None)
    monkeypatch.setattr(gemini_client.config, "GEMINI_API_KEY", "test-key")

    seen = []
    threads = [threading.Thread(target=lambda: seen.append(gemini_client.get_client())) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(built) == 1 and len({id(c) for c in seen}) == 1 and len(seen) == 8


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(gemini_client, "_client", None)
    monkeypatch.setattr(gemini_client.config, "GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_client.get_client()


def test_is_transient_classification():
    assert is_transient(httpx.ReadError("x")) and is_transient(api_error(500))
    assert not is_transient(api_error(404)) and not is_transient(RuntimeError("x"))
