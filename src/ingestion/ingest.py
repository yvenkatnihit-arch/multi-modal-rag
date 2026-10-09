"""Ingestion: data/<topic>/ files -> Records -> Chunks -> embeddings -> Chroma (one collection per topic).

    python -m src.ingestion.ingest                 # all topics
    python -m src.ingestion.ingest receipts song_lyrics

A file that fails to parse is logged and skipped, and its already-stored chunks are protected
from deletion, so a transient error can never erase good data.
"""
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from src import config  # noqa: F401  (before chromadb)
from src.core.assets import prune_tables
from src.ingestion.chunker import chunk_records
from src.ingestion.file_router import route_topic
from src.ingestion.parsers.pdf_parser import parse_pdf
from src.ingestion.parsers.image_parser import parse_image
from src.ingestion.parsers.table_parser import parse_table
from src.ingestion.parsers.text_parser import parse_text
from src.core.topic_registry import Topic, list_topics
from src.core.vector_store import SyncStats, VectorStore

log = logging.getLogger(__name__)

PARSERS = {"text": parse_text, "pdf": parse_pdf, "table": parse_table, "image": parse_image}


@dataclass
class TopicReport:
    topic_id: str
    files: int = 0
    records: int = 0
    chunks: int = 0
    failed: dict[str, str] = field(default_factory=dict)      # source -> error
    no_parser: dict[str, int] = field(default_factory=dict)   # kind -> file count
    skipped: int = 0
    stats: SyncStats | None = None
    seconds: float = 0.0


def ingest_topic(topic: Topic, store: VectorStore) -> TopicReport:
    t0 = time.time()
    report = TopicReport(topic.id)
    routed, skipped = route_topic(topic)
    report.skipped = len(skipped)

    def run(f):
        try:
            return f, PARSERS[f.kind](f, topic.id), None
        except Exception as e:                                # one bad file must not stop the topic
            log.error("[%s] failed to parse %s: %s: %s", topic.id, f.source, type(e).__name__, e)
            return f, [], f"{type(e).__name__}: {e}"

    runnable = [f for f in routed if f.kind in PARSERS]
    for f in routed:
        if f.kind not in PARSERS:
            report.no_parser[f.kind] = report.no_parser.get(f.kind, 0) + 1
    report.files = len(runnable)

    # Images wait on the network (one vision call each), so they run a few at a time; results keep file order.
    slow = [f for f in runnable if f.kind == "image"]
    with ThreadPoolExecutor(max_workers=config.IMAGE_WORKERS) as pool:
        slow_results = dict(zip((f.source for f in slow), pool.map(run, slow)))
    records = []
    for f in runnable:
        _, recs, error = slow_results[f.source] if f.kind == "image" else run(f)
        if error:
            report.failed[f.source] = error
        records += recs

    if not report.failed:                  # only after a clean run: a failed file must keep its old assets
        keep = {(r.metadata.source, r.metadata.table_id) for r in records if r.metadata.modality == "table"}
        removed = prune_tables(topic.id, keep)
        if removed:
            log.info("[%s] removed %d stale table asset(s): %s", topic.id, len(removed), removed)

    chunks = chunk_records(records)
    report.records, report.chunks = len(records), len(chunks)
    report.stats = store.sync_topic(
        topic.id, chunks,
        protect_sources=report.failed.keys(),
        allow_empty=not report.failed and not report.no_parser,   # an all-pending topic is not "really empty"
    )
    report.seconds = time.time() - t0
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("topics", nargs="*", help="topic ids (default: all)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config.ensure_dirs()
    available = {t.id: t for t in list_topics()}
    unknown = [t for t in args.topics if t not in available]
    if unknown:
        print(f"unknown topic(s): {unknown}. available: {sorted(available)}")
        return 2
    selected = [available[t] for t in args.topics] if args.topics else list(available.values())

    store = VectorStore()
    failures = 0
    for topic in selected:
        r = ingest_topic(topic, store)
        failures += len(r.failed)
        pending = f" | awaiting parser: {r.no_parser}" if r.no_parser else ""
        print(f"{topic.id:18} files={r.files:3} chunks={r.chunks:4} -> {r.stats}  ({r.seconds:.1f}s){pending}")
        for src, err in r.failed.items():
            print(f"    FAILED {src}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
