"""
Face sync — polls cloud backend for face embedding updates.

Every POLL_INTERVAL seconds:
  1. GET /api/v1/faces/sync?since=<last_sync_at>
  2. Apply add/delete actions to local SQLite
  3. Decode + decrypt embeddings
  4. Trigger recognizer reload via callback

Handles offline gracefully — just waits and retries.
"""
import base64
import logging
import struct
import time
from datetime import datetime, timezone

import numpy as np
import requests

from local_db.db import get_conn

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 10
_BACKOFF_MAX     = 600   # 10 min max backoff on server errors


def _decode_embedding(b64_data: str) -> np.ndarray:
    """Decode base64 float32 embedding bytes → numpy array."""
    raw = base64.b64decode(b64_data)
    n = len(raw) // 4
    return np.array(struct.unpack(f"{n}f", raw), dtype=np.float32)


class FaceSync:
    """
    Keeps local face DB in sync with cloud.

    Usage:
        sync = FaceSync(BACKEND_URL, AGENT_TOKEN, DB_PATH, on_sync=recognizer.reload_persons)
        threading.Thread(target=sync.run, daemon=True).start()
    """

    POLL_INTERVAL = 60  # seconds between sync polls

    def __init__(self, cloud_base_url: str, agent_token: str,
                 db_path: str, on_sync=None):
        self.cloud_base_url = cloud_base_url.rstrip("/")
        self.agent_token = agent_token
        self.db_path = db_path
        self.on_sync = on_sync  # callable — called after each successful sync
        self._backoff = self.POLL_INTERVAL

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self):
        """Run sync loop forever. Run this in a daemon thread."""
        logger.info("FaceSync: started")
        while True:
            try:
                added, deleted = self._sync()
                if added + deleted > 0:
                    logger.info(f"FaceSync: +{added} added, -{deleted} deleted")
                    if self.on_sync:
                        self.on_sync()
                self._backoff = self.POLL_INTERVAL  # reset on success
            except requests.exceptions.ConnectionError:
                logger.debug("FaceSync: offline — will retry later")
                self._backoff = min(self._backoff * 2, _BACKOFF_MAX)
            except Exception as e:
                logger.error(f"FaceSync error: {e}")
                self._backoff = min(self._backoff * 2, _BACKOFF_MAX)
            time.sleep(self._backoff)

    def force_sync(self):
        """Trigger an immediate sync (blocking). Useful at startup."""
        try:
            added, deleted = self._sync()
            logger.info(f"FaceSync force_sync: +{added} added, -{deleted} deleted")
            if (added + deleted > 0) and self.on_sync:
                self.on_sync()
        except Exception as e:
            logger.warning(f"FaceSync force_sync failed: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sync(self) -> tuple[int, int]:
        """
        Fetch actions from cloud, apply to local DB.
        Returns (added_count, deleted_count).
        """
        last_sync = self._get_last_sync()
        params = {}
        if last_sync:
            params["since"] = last_sync

        resp = requests.get(
            f"{self.cloud_base_url}/api/v1/faces/sync",
            params=params,
            headers={"Authorization": f"Bearer {self.agent_token}"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        actions = data.get("actions", [])
        synced_at = data.get("synced_at", datetime.now(timezone.utc).isoformat())

        added, deleted = 0, 0
        for action in actions:
            try:
                if action["action"] == "add":
                    self._apply_add(action)
                    added += 1
                elif action["action"] == "delete":
                    self._apply_delete(action["id"])
                    deleted += 1
            except Exception as e:
                logger.warning(f"FaceSync: failed to apply action {action}: {e}")

        self._save_last_sync(synced_at)
        return added, deleted

    def _apply_add(self, action: dict):
        """
        Add or update person in local SQLite.

        Expected action fields:
          id, name, type, embedding (base64 float32 bytes), updated_at
        """
        person_id  = action["id"]
        name       = action["name"]
        ptype      = action["type"]
        updated_at = action.get("updated_at", datetime.now(timezone.utc).isoformat())

        # Decode embedding
        emb_b64 = action.get("embedding")
        if not emb_b64:
            logger.warning(f"FaceSync: person {person_id} has no embedding — skipping")
            return

        embedding = _decode_embedding(emb_b64)
        blob = embedding.astype(np.float32).tobytes()

        conn = get_conn(self.db_path)
        try:
            conn.execute(
                """INSERT INTO persons (id, name, type, embedding, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, type=excluded.type,
                     embedding=excluded.embedding,
                     updated_at=excluded.updated_at""",
                (person_id, name, ptype, blob, updated_at),
            )
            conn.commit()
            logger.debug(f"FaceSync: upserted person {person_id} ({name}, {ptype})")
        finally:
            conn.close()

    def _apply_delete(self, person_id: str):
        conn = get_conn(self.db_path)
        try:
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            conn.commit()
            logger.debug(f"FaceSync: deleted person {person_id}")
        finally:
            conn.close()

    def _get_last_sync(self) -> str | None:
        conn = get_conn(self.db_path)
        try:
            row = conn.execute(
                "SELECT value FROM sync_state WHERE key = 'last_sync_at'"
            ).fetchone()
            return row["value"] if row else None
        finally:
            conn.close()

    def _save_last_sync(self, synced_at: str):
        conn = get_conn(self.db_path)
        try:
            conn.execute(
                """INSERT INTO sync_state (key, value)
                   VALUES ('last_sync_at', ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (synced_at,),
            )
            conn.commit()
        finally:
            conn.close()
