"""
Face recognizer — matches detected embeddings against local SQLite person DB.
Fires the right event type based on person role and whether they're known.
"""
import logging
import sqlite3
import struct

import numpy as np

from local_db.db import get_conn

logger = logging.getLogger(__name__)

# Event types per role
_ROLE_EVENT = {
    "vip":         "vip_spotted",
    "blacklisted": "blacklist_alert",
    "staff":       "checkin",
}


def _bytes_to_embedding(blob: bytes) -> np.ndarray:
    n = len(blob) // 4  # float32 = 4 bytes
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    arr = embedding.astype(np.float32)
    return struct.pack(f"{len(arr)}f", *arr)


class Recognizer:
    """
    Matches face embeddings against locally-cached person database.

    Thread safety: loads persons into memory at startup and after sync.
    The _persons list is replaced atomically (single assignment) so
    concurrent reads during a reload are safe without a lock.
    """

    def __init__(self, db_path: str, confidence_threshold: float = 0.95):
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold
        self._persons: list[dict] = []  # {id, name, type, embedding (np.ndarray)}

    # ── Public API ────────────────────────────────────────────────────────────

    def reload_persons(self):
        """
        Reload face embeddings from local SQLite.
        Called once at startup and again after each face sync.
        """
        conn = get_conn(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id, name, type, embedding FROM persons"
            ).fetchall()
        finally:
            conn.close()

        persons = []
        for row in rows:
            try:
                emb = _bytes_to_embedding(row["embedding"])
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm  # pre-normalise for fast cosine sim
                persons.append({
                    "id":        row["id"],
                    "name":      row["name"],
                    "type":      row["type"],
                    "embedding": emb,
                })
            except Exception as e:
                logger.warning(f"Skipping person {row['id']}: bad embedding — {e}")

        self._persons = persons  # atomic replacement
        logger.info(f"Recognizer: loaded {len(self._persons)} persons from DB")

    def match(self, embedding: np.ndarray) -> dict | None:
        """
        Compare embedding against all known persons using cosine similarity.

        Returns:
            {person_id, name, role, confidence, event_type}
            or None if no match above threshold.
        """
        if not self._persons:
            return None

        # Normalise query embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        query = embedding / norm

        best_sim = -1.0
        best_person = None

        for person in self._persons:
            sim = float(np.dot(query, person["embedding"]))
            if sim > best_sim:
                best_sim = sim
                best_person = person

        if best_sim < self.confidence_threshold:
            logger.debug(
                f"No match above threshold {self.confidence_threshold:.2f} "
                f"(best: {best_sim:.3f})"
            )
            return None

        role = best_person["type"]
        event_type = _ROLE_EVENT.get(role, "checkin")

        logger.info(
            f"Matched: {best_person['name']} ({role}) "
            f"confidence={best_sim:.3f} event={event_type}"
        )
        return {
            "person_id":  best_person["id"],
            "name":       best_person["name"],
            "role":       role,
            "confidence": best_sim,
            "event_type": event_type,
        }

    def identify_or_unknown(self, embedding: np.ndarray) -> dict:
        """
        Like match() but always returns a result.
        Returns unknown_face event if no match above threshold.
        """
        result = self.match(embedding)
        if result:
            return result
        return {
            "person_id":  None,
            "name":       None,
            "role":       None,
            "confidence": None,
            "event_type": "unknown_face",
        }

    def add_person(self, person_id: str, name: str, person_type: str,
                   embedding: np.ndarray):
        """
        Add or update a person in the local DB and in-memory list.
        Used by face_sync after a successful cloud sync.
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        blob = _embedding_to_bytes(embedding)

        conn = get_conn(self.db_path)
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO persons (id, name, type, embedding, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, type=excluded.type,
                     embedding=excluded.embedding, updated_at=excluded.updated_at""",
                (person_id, name, person_type, blob, now),
            )
            conn.commit()
        finally:
            conn.close()

    def remove_person(self, person_id: str):
        """Remove a person from local DB. Used by face_sync for delete actions."""
        conn = get_conn(self.db_path)
        try:
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            conn.commit()
        finally:
            conn.close()
