"""Read everything in data/<topic>/ into the search index (and assets/).

    python ingest.py                     all topics
    python ingest.py receipts song_lyrics

Safe to re-run: unchanged files cost nothing. The same thing as  python -m src.ingestion.ingest
"""
import sys

from src.ingestion.ingest import main

if __name__ == "__main__":
    sys.exit(main())
