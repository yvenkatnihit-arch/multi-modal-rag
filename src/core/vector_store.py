"""Vector store: one ChromaDB collection per topic, kept in sync with the chunks (never rebuilt).

sync_topic() compares what is stored with what the chunker produced:
  new / changed chunks  -> embedded and upserted
  unchanged chunks      -> untouched (no API call)
  chunks that vanished  -> deleted
"""
import hashlib
import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from src import config  # noqa: F401  (must come before chromadb: installs the gRPC telemetry stub)

import chromadb

from src.core.embeddings import embed_documents
from src.core.records import Chunk

log = logging.getLogger(__name__)

WRITE_BATCH = 500
HASH_KEY = "content_hash"


@dataclass(frozen=True)
class SyncStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    protected: int = 0       # stale chunks kept because their source failed to parse this run

    def __str__(self) -> str:
        return (f"+{self.added} added, ~{self.updated} updated, ={self.unchanged} unchanged, "
                f"-{self.deleted} deleted" + (f", {self.protected} protected" if self.protected else ""))


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict
    distance: float | None = None     # cosine distance: 0 = identical, 2 = opposite


def collection_name(topic_id: str) -> str:
    return f"topic_{topic_id}"        # Chroma needs 3+ chars; the prefix guarantees it


class VectorStore:
    def __init__(self, path=None, embed_fn: Callable[[list[str]], list[list[float]]] = embed_documents):
        self._client = chromadb.PersistentClient(path=str(path or config.CHROMA_DIR))
        self._embed = embed_fn

    # ------------------------------------------------------------ collections
    def collection(self, topic_id: str):
        return self._client.get_or_create_collection(
            collection_name(topic_id), metadata={"hnsw:space": "cosine"}
        )

    def topics(self) -> list[str]:
        return sorted(c.name.removeprefix("topic_") for c in self._client.list_collections())

    def count(self, topic_id: str) -> int:
        return self.collection(topic_id).count()

    def fingerprint(self, topic_id: str) -> str:
        """Cheap identity of a topic's current contents (reads metadata only, no documents).
        Changes whenever any chunk is added, removed or modified, so caches can tell when to rebuild."""
        got = self.collection(topic_id).get(include=["metadatas"])
        h = hashlib.sha1()
        for chunk_id, meta in sorted(zip(got["ids"], got["metadatas"]), key=lambda p: p[0]):
            h.update(f"{chunk_id}:{(meta or {}).get(HASH_KEY, '')};".encode("utf-8"))
        return h.hexdigest()

    def delete_topic(self, topic_id: str) -> None:
        """Explicit, deliberate removal of a whole topic (sync_topic never does this)."""
        self._client.delete_collection(collection_name(topic_id))

    # ------------------------------------------------------------ writing
    def sync_topic(
        self,
        topic_id: str,
        chunks: Iterable[Chunk],
        protect_sources: Iterable[str] = (),
        allow_empty: bool = False,
    ) -> SyncStats:
        chunks = list(chunks)
        col = self.collection(topic_id)
        stored = col.get(include=["metadatas"])
        stored_hash = {i: (m or {}).get(HASH_KEY) for i, m in zip(stored["ids"], stored["metadatas"])}
        stored_source = {i: (m or {}).get("source") for i, m in zip(stored["ids"], stored["metadatas"])}

        if not chunks and stored_hash and not allow_empty:
            raise ValueError(
                f"refusing to wipe topic {topic_id!r} ({len(stored_hash)} chunks) with an empty chunk list; "
                "pass allow_empty=True if the topic really is empty"
            )
        if any(c.metadata.topic_id != topic_id for c in chunks):
            raise ValueError(f"chunks from another topic passed to sync_topic({topic_id!r})")

        new = [c for c in chunks if c.id not in stored_hash]
        changed = [c for c in chunks if c.id in stored_hash and stored_hash[c.id] != c.content_hash]
        unchanged = len(chunks) - len(new) - len(changed)

        keep = set(protect_sources)
        current_ids = {c.id for c in chunks}
        stale = [i for i in stored_hash if i not in current_ids]
        protected = [i for i in stale if stored_source.get(i) in keep]
        stale = [i for i in stale if i not in set(protected)]

        to_write = new + changed
        for start in range(0, len(to_write), WRITE_BATCH):
            batch = to_write[start : start + WRITE_BATCH]
            vectors = self._embed([c.text for c in batch])
            col.upsert(
                ids=[c.id for c in batch],
                embeddings=vectors,
                documents=[c.text for c in batch],
                metadatas=[{**c.metadata.to_dict(), HASH_KEY: c.content_hash} for c in batch],
            )
        for start in range(0, len(stale), WRITE_BATCH):
            col.delete(ids=stale[start : start + WRITE_BATCH])

        stats = SyncStats(len(new), len(changed), unchanged, len(stale), len(protected))
        log.info("[%s] %s", topic_id, stats)
        return stats

    # ------------------------------------------------------------ reading
    def dense_search(self, topic_id: str, query_embedding: list[float], k: int, modality: str | None = None) -> list[Hit]:
        col = self.collection(topic_id)
        if col.count() == 0:
            return []
        res = col.query(
            query_embeddings=[query_embedding], n_results=min(k, col.count()),
            where={"modality": modality} if modality else None,
            include=["documents", "metadatas", "distances"],
        )
        return [Hit(i, d, m, dist) for i, d, m, dist in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def get_all(self, topic_id: str, modality: str | None = None) -> list[Hit]:
        """Every chunk of a topic (used to build the BM25 keyword index)."""
        res = self.collection(topic_id).get(where={"modality": modality} if modality else None,
                                            include=["documents", "metadatas"])
        return [Hit(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
