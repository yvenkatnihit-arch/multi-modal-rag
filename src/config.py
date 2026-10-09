"""Central configuration: every path, model name and tunable lives here."""
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

# Windows Application Control blocks grpcio's compiled DLL on this machine.
# ChromaDB imports the gRPC OpenTelemetry exporter only for optional telemetry,
# so we register a stand-in module before chromadb is ever imported.
_stub = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
_stub.OTLPSpanExporter = object
sys.modules.setdefault(_stub.__name__, _stub)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Paths ---
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
ASSETS_DIR = ROOT / "assets"
LOGS_DIR = ROOT / "logs"

# --- Models ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TABLE_MAX_CHARS = 3000       # table records longer than this are split by lines (header repeated)
IMAGE_MAX_CHARS = 3000       # caption/OCR text longer than this is split like text
HEADING_PREFIX_MAX = 200     # cap on the heading path repeated at the top of each text chunk

# --- Query side ---
ROUTER_MAX_TOPICS = 3        # the router never sends a question to more topics than this
HISTORY_MAX_MESSAGES = 6     # how much chat history the rewriter sees
HISTORY_MAX_CHARS = 600      # each past message is cut to this length (answers can be long)

CONTEXT_MAX_TOKENS = 6000    # how much retrieved text goes into the answer prompt (estimated)
CHARS_PER_TOKEN = 3.5        # rough estimate, deliberately on the cautious side for numbers and tables
GENERATOR_THINKING_BUDGET = 1024   # a little reasoning helps when reading tables and OCR text
MAX_PICTURES_TO_LLM = 3      # the answer step looks at no more than this many original pictures per question

# --- Images ---
IMAGE_WORKERS = 3            # vision calls run this many at a time during ingestion

# --- Tables ---
TABLE_EMBED_ROWS = 500       # tables with more rows than this are sampled for row-level search
TABLE_GROUP_CHARS = 1000     # target size of one row-group record
TABLE_CELL_MAX = 150         # longest cell text shown in row groups (the stored table is untouched)
TABLE_ENUM_MAX = 15          # a column with at most this many distinct values lists them all
TABLE_SAMPLE_SEED = 42

# --- Retrieval ---
TOP_K = 6                    # chunks kept per topic search, and in the merged list
MAX_QUESTION_CHARS = 2000    # longer questions are rejected before any API call


def ensure_dirs() -> None:
    for d in (DATA_DIR, CHROMA_DIR, ASSETS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
