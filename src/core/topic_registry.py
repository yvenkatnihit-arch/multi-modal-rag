"""Topic registry: every sub-folder of data/ is a topic; folder name = topic id.

Each topic may carry a topic.json {"name": ..., "description": ...}. The description
tells the router what kinds of data the topic holds.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src import config

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[a-z0-9_]+$")  # becomes a Chroma collection name, so keep it simple


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    description: str
    path: Path


def _load_topic(folder: Path) -> Topic | None:
    if not _ID_RE.match(folder.name):
        log.warning("Skipping folder %r: topic ids must be lowercase letters, digits, underscores", folder.name)
        return None
    name, description = folder.name.replace("_", " ").title(), ""
    meta = folder / "topic.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8-sig"))  # tolerate a BOM
            name = data.get("name", name)
            description = data.get("description", "")
        except json.JSONDecodeError as e:
            log.warning("Bad topic.json in %s: %s", folder.name, e)
    if not description:
        log.warning("Topic %r has no description; the router will route poorly to it", folder.name)
        description = f"Documents about {name.lower()}."
    return Topic(id=folder.name, name=name, description=description, path=folder)


def list_topics(data_dir: Path | None = None) -> list[Topic]:
    root = data_dir or config.DATA_DIR
    if not root.exists():
        return []
    topics = (_load_topic(f) for f in sorted(root.iterdir()) if f.is_dir())
    return [t for t in topics if t is not None]


def get_topic(topic_id: str) -> Topic:
    for t in list_topics():
        if t.id == topic_id:
            return t
    raise KeyError(f"Unknown topic: {topic_id}")


if __name__ == "__main__":
    for t in list_topics():
        print(f"{t.id:18} {t.name:18} {t.description[:70]}...")
