"""
Event uploader — POSTs recognition events to cloud backend.

Stores events in SQLite pending_events table when offline.
A background thread drains the queue whenever internet is available.
Retries with exponential backoff on failure.
"""
import logging
import time
import uuid
from datetime import datetime, timezone

import requests

from local_db.db import get_conn

logger = logging.getLogger(__name__)

_UPLOAD_INTERVAL = 5       # seconds between drain attempts
_MAX_BATCH        = 20     # events per upload batch
_BACKOFF_MAX      = 300    # max seconds between retries (5 min)
_REQUEST_TIMEOUT  = 10     # seconds


class EventUploader:
    """
    Reliably uploads recognition events to cloud with offline queue.

    Usage:
        uploader = EventUploader(BACKEND_URL, AGENT_TOKEN, DB_PATH)
        uploader.enqueue(event_dict)   # call from main thread
        threading.Thread(target=uploader.run, daemon=True).start()
    """

    def __init__(self, cloud_base_url: str, agent_token: str,
                 db_path: str, site_id: str):
        self.cloud_base_url = cloud_base_url.rstrip("/")
        self.agent_token = agent_token
        self.db_path = db_path
        self.site_id = site_id
        self._backoff = _UPLOAD_INTERVAL

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(self, event: dict):
        """
        Persist event to SQLite offline queue.
        Never raises — if DB write fails, logs and continues.

        event keys: camera_id, person_id, person_name, role, confidence,
                    event_type, timestamp, frame_path (optional)
        """
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = get_conn(self.db_path)
        try:
            conn.execute(
                """INSERT INTO pending_events
                   (id, camera_id, person_id, confidence, event_type,
                    timestamp, clip_path, created_at, uploaded)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    event_id,
                    event["camera_id"],
                    event.get("person_id"),
                    event.get("confidence"),
                    event["event_type"],
                    event.get("timestamp", now),
                    event.get("frame_path"),
                    now,
                ),
            )
            conn.commit()
            logger.debug(f"Enqueued event {event_id} ({event['event_type']})")
        except Exception as e:
            logger.error(f"Failed to enqueue event: {e}")
        finally:
            conn.close()

    def run(self):
        """Drain upload queue forever. Run this in a daemon thread."""
        logger.info("EventUploader: started")
        while True:
            try:
                uploaded = self._drain_batch()
                if uploaded > 0:
                    self._backoff = _UPLOAD_INTERVAL  # reset on success
            except Exception as e:
                logger.error(f"EventUploader drain error: {e}")
                self._backoff = min(self._backoff * 2, _BACKOFF_MAX)
            time.sleep(self._backoff)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _drain_batch(self) -> int:
        """
        Fetch up to _MAX_BATCH un-uploaded events, POST them, mark as uploaded.
        Returns number of events successfully uploaded.
        """
        conn = get_conn(self.db_path)
        try:
            rows = conn.execute(
                """SELECT id, camera_id, person_id, confidence,
                          event_type, timestamp, clip_path
                   FROM pending_events
                   WHERE uploaded = 0
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (_MAX_BATCH,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return 0

        uploaded_ids = []
        for row in rows:
            payload = {
                "camera_id":   row["camera_id"],
                "person_id":   row["person_id"],
                "confidence":  row["confidence"],
                "event_type":  row["event_type"],
                "timestamp":   row["timestamp"],
                "frame_path":  row["clip_path"],
                "site_id":     self.site_id,
            }
            try:
                resp = requests.post(
                    f"{self.cloud_base_url}/api/v1/events",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.agent_token}"},
                    timeout=_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                uploaded_ids.append(row["id"])
                logger.debug(
                    f"Uploaded event {row['id']} ({row['event_type']}) → {resp.status_code}"
                )
            except requests.exceptions.ConnectionError:
                logger.debug("EventUploader: offline — will retry later")
                break  # stop trying this batch, wait for next interval
            except requests.exceptions.HTTPError as e:
                if resp.status_code in (400, 422):
                    # Bad payload — mark as uploaded to avoid infinite retry
                    logger.warning(
                        f"Event {row['id']} rejected by server ({resp.status_code}) — discarding"
                    )
                    uploaded_ids.append(row["id"])
                else:
                    logger.warning(f"Server error uploading event {row['id']}: {e}")
                    break
            except Exception as e:
                logger.warning(f"Unexpected error uploading event {row['id']}: {e}")
                break

        if uploaded_ids:
            self._mark_uploaded(uploaded_ids)

        return len(uploaded_ids)

    def _mark_uploaded(self, event_ids: list[str]):
        placeholders = ",".join("?" * len(event_ids))
        conn = get_conn(self.db_path)
        try:
            conn.execute(
                f"UPDATE pending_events SET uploaded=1 WHERE id IN ({placeholders})",
                event_ids,
            )
            conn.commit()
            logger.info(f"EventUploader: marked {len(event_ids)} event(s) as uploaded")
        except Exception as e:
            logger.error(f"Failed to mark events as uploaded: {e}")
        finally:
            conn.close()

    def pending_count(self) -> int:
        """Return number of events waiting to be uploaded (for health checks)."""
        conn = get_conn(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) as n FROM pending_events WHERE uploaded=0"
            ).fetchone()
            return row["n"] if row else 0
        finally:
            conn.close()
