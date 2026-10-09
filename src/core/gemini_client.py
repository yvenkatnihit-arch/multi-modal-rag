"""One shared Gemini client plus retry-with-backoff for transient failures."""
import logging
import threading
import time
from typing import Callable, TypeVar

from src import config

log = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)   # silences the harmless "automatic function calling" notice

T = TypeVar("T")
MAX_RETRIES = 6
RETRY_CODES = {429, 500, 502, 503, 504}

_client = None
_client_lock = threading.Lock()


def get_client():
    """The one shared client. Created under a lock: several threads (parallel image / page reads) can ask for it
    at the same moment on first use, and a client that gets replaced is closed while another thread may be using it."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not config.GEMINI_API_KEY:
                    raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
                from google import genai
                _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def is_transient(e: BaseException) -> bool:
    """Worth retrying: the API said 'busy / try again' (429, 5xx), or the connection itself failed
    (dropped, reset, timed out). Anything else (bad request, blocked content, a bug) is not retried."""
    import httpx
    from google.genai import errors

    if isinstance(e, errors.APIError):
        return e.code in RETRY_CODES
    return isinstance(e, (httpx.TransportError, ConnectionError, TimeoutError))


def call_with_retries(fn: Callable[[], T], what: str = "Gemini call") -> T:
    """Run fn(); on a transient failure wait 2, 4, 8... seconds (max 60) and try again."""
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES or not is_transient(e):
                raise
            log.warning("%s failed (%s: %s), retry %d/%d in %.0fs", what, type(e).__name__, e, attempt, MAX_RETRIES, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise AssertionError("unreachable")
