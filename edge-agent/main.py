"""
VisionEdge Edge Agent — Entry point.
Runs on Jetson Nano / Mini PC at client site.

Threads:
  - StreamManager       : one thread per camera, captures RTSP frames
  - Recognition loop    : reads latest frames, detects + matches faces, enqueues events
  - FaceSync            : polls cloud every 60s for DB updates
  - EventUploader       : drains SQLite queue → cloud POST every 5s
"""
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from config import load_config
from local_db.db import init_db
from camera.stream_manager import StreamManager
from ai.face_detector import FaceDetector
from ai.recognizer import Recognizer
from sync.face_sync import FaceSync
from sync.event_uploader import EventUploader

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("edge-agent")

# How often the recognition loop processes a frame per camera (seconds)
_RECOGNITION_INTERVAL = 1.0


# Cooldown: don't re-fire same person within this window (seconds)
# checkin is handled separately via attendance_log (once per day)
_COOLDOWN_SECONDS = {
    "blacklist_alert": 10,
    "vip_spotted":     30,
    "unknown_face":    30,
}
_DEFAULT_COOLDOWN = 30


def _already_checked_in_today(db_path: str, person_id: str) -> bool:
    """Return True if this person already has an attendance record for today (UTC)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM attendance_log WHERE person_id = ? AND date = ?",
            (person_id, today),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _record_attendance(db_path: str, person_id: str, camera_id: str, timestamp: str):
    """Insert attendance record for today. Ignored if already exists (IGNORE)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_conn(db_path)
    try:
        conn.execute(
            """INSERT OR IGNORE INTO attendance_log
               (person_id, date, first_camera, checked_in_at)
               VALUES (?, ?, ?, ?)""",
            (person_id, today, camera_id, timestamp),
        )
        conn.commit()
    finally:
        conn.close()


# Anti-spoof: staff must be recognised this many times within the window
# before attendance is logged.  Defeats a quickly-flashed phone photo.
_LIVENESS_FRAMES_REQUIRED = 3
_LIVENESS_WINDOW_SECONDS  = 10


def _process_camera(
    camera_id: str,
    stream_manager: StreamManager,
    detector: FaceDetector,
    recognizer: Recognizer,
    uploader: EventUploader,
    db_path: str,
    last_seen: dict,           # (person_id_or_key, camera_id) → last event timestamp
    last_seen_lock: threading.Lock,
    liveness_buf: dict,        # person_id → {"count": int, "first_seen": float}
    liveness_lock: threading.Lock,
):
    """
    Process one camera frame: detect → liveness check → match → dedup → enqueue.

    Anti-spoof layers for staff check-in:
      1. Laplacian variance — rejects flat photos/phone screens (low texture).
      2. Multi-frame confirmation — person must appear in 3 frames within 10s
         before attendance is recorded.  A quickly-flashed photo fails this.
    """
    frame = stream_manager.get_frame(camera_id)
    if frame is None:
        return

    faces = detector.detect(frame, enhance=True)
    now = time.time()
    ts  = datetime.now(timezone.utc).isoformat()

    for face in faces:
        result     = recognizer.identify_or_unknown(face["embedding"])
        event_type = result["event_type"]
        person_id  = result["person_id"]

        if event_type == "checkin" and person_id:
            # ── Layer 1: texture liveness ──────────────────────────────────
            if face["liveness"] < detector.LIVENESS_THRESHOLD:
                logger.debug(
                    f"[{camera_id}] Liveness check FAILED for {result['name']} "
                    f"(score={face['liveness']:.1f} < {detector.LIVENESS_THRESHOLD}) "
                    f"— likely photo/screen"
                )
                continue  # reject — too flat, probably a printed photo or phone screen

            # ── Already checked in today? ──────────────────────────────────
            if _already_checked_in_today(db_path, person_id):
                continue

            # ── Layer 2: multi-frame confirmation ──────────────────────────
            # Person must be seen LIVENESS_FRAMES_REQUIRED times within
            # LIVENESS_WINDOW_SECONDS before we trust it's a live face.
            with liveness_lock:
                entry = liveness_buf.get(person_id)
                if entry is None or (now - entry["first_seen"]) > _LIVENESS_WINDOW_SECONDS:
                    # Start / reset window
                    liveness_buf[person_id] = {"count": 1, "first_seen": now}
                    logger.debug(f"[{camera_id}] Anti-spoof: {result['name']} 1/{_LIVENESS_FRAMES_REQUIRED}")
                    continue  # need more frames
                else:
                    entry["count"] += 1
                    logger.debug(
                        f"[{camera_id}] Anti-spoof: {result['name']} "
                        f"{entry['count']}/{_LIVENESS_FRAMES_REQUIRED}"
                    )
                    if entry["count"] < _LIVENESS_FRAMES_REQUIRED:
                        continue  # not enough confirmations yet
                    # Confirmed — remove from buffer and proceed
                    del liveness_buf[person_id]

            # Both layers passed — record attendance and upload once
            _record_attendance(db_path, person_id, camera_id, ts)

        else:
            # VIP / blacklist / unknown: time-based cooldown per (person, camera)
            person_key   = person_id or "__unknown__"
            cooldown_key = (person_key, camera_id)
            cooldown     = _COOLDOWN_SECONDS.get(event_type, _DEFAULT_COOLDOWN)

            with last_seen_lock:
                if now - last_seen.get(cooldown_key, 0) < cooldown:
                    continue
                last_seen[cooldown_key] = now

        event = {
            "camera_id":    camera_id,
            "person_id":    person_id,
            "person_name":  result["name"],
            "role":         result["role"],
            "confidence":   result["confidence"],
            "event_type":   event_type,
            "timestamp":    ts,          # exact check-in time included here
            "frame_path":   None,
        }
        uploader.enqueue(event)
        logger.info(
            f"[{camera_id}] {event_type} "
            f"person={result['name'] or 'unknown'} "
            f"conf={result['confidence']} "
            f"liveness={face['liveness']:.1f} "
            f"time={ts}"
        )


def recognition_loop(
    stream_manager: StreamManager,
    camera_ids: list[str],
    detector: FaceDetector,
    recognizer: Recognizer,
    uploader: EventUploader,
    db_path: str,
    stop_event: threading.Event,
):
    """
    Main AI loop — runs in its own thread.
    Processes all cameras in parallel (one worker per camera).
    Staff check-in: once per day via attendance_log (survives restarts).
    VIP/blacklist/unknown: time-based cooldown per (person, camera).
    """
    logger.info("Recognition loop: started")
    last_seen: dict[tuple, float] = {}    # (person_id_or_key, camera_id) → epoch time
    last_seen_lock  = threading.Lock()
    liveness_buf: dict[str, dict] = {}    # person_id → {count, first_seen}
    liveness_lock   = threading.Lock()
    max_workers = max(len(camera_ids), 1)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cam-worker") as pool:
        while not stop_event.is_set():
            futures = {
                pool.submit(
                    _process_camera,
                    camera_id, stream_manager, detector, recognizer,
                    uploader, db_path, last_seen, last_seen_lock,
                    liveness_buf, liveness_lock,
                ): camera_id
                for camera_id in camera_ids
            }
            for future in as_completed(futures):
                camera_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Recognition error on camera {camera_id}: {e}")

            stop_event.wait(_RECOGNITION_INTERVAL)

    logger.info("Recognition loop: stopped")


def main():
    config = load_config()
    logger.info(f"Starting VisionEdge Edge Agent — site_id={config.site_id}")

    db_path = os.environ.get("DB_PATH", "/var/lib/visionedge/edge-agent.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # ── Database ──────────────────────────────────────────────────────────────
    logger.info(f"Initialising database: {db_path}")
    init_db(db_path)

    # ── AI components ─────────────────────────────────────────────────────────
    det_thresh = float(os.environ.get("INSIGHTFACE_DET_THRESH", "0.3"))
    ctx_id = int(os.environ.get("INSIGHTFACE_CTX_ID", "-1"))

    detector = FaceDetector(det_thresh=det_thresh)
    detector.load()
    logger.info("FaceDetector: loaded buffalo_l")

    recognizer = Recognizer(
        db_path=db_path,
        confidence_threshold=config.confidence_threshold,
    )

    # ── Sync components ───────────────────────────────────────────────────────
    uploader = EventUploader(
        cloud_base_url=config.cloud_base_url,
        agent_token=config.agent_token,
        db_path=db_path,
        site_id=config.site_id,
    )

    face_sync = FaceSync(
        cloud_base_url=config.cloud_base_url,
        agent_token=config.agent_token,
        db_path=db_path,
        on_sync=recognizer.reload_persons,
    )

    # Initial sync + load before starting recognition
    logger.info("FaceSync: performing initial sync...")
    face_sync.force_sync()
    recognizer.reload_persons()
    logger.info(f"Recognizer: ready with {len(recognizer._persons)} persons")

    # ── Camera streams ────────────────────────────────────────────────────────
    stream_manager = StreamManager(cameras=config.cameras)
    stream_manager.start()
    camera_ids = [cam["camera_id"] for cam in config.cameras]
    logger.info(f"StreamManager: started {len(config.cameras)} camera(s)")

    # ── Background threads ────────────────────────────────────────────────────
    stop_event = threading.Event()

    threads = [
        threading.Thread(
            target=uploader.run,
            name="event-uploader",
            daemon=True,
        ),
        threading.Thread(
            target=face_sync.run,
            name="face-sync",
            daemon=True,
        ),
        threading.Thread(
            target=recognition_loop,
            args=(stream_manager, camera_ids, detector, recognizer, uploader, db_path, stop_event),
            name="recognition-loop",
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()
        logger.info(f"Thread '{t.name}' started")

    logger.info("Edge Agent running. Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_event.set()
        stream_manager.stop()
        logger.info("Edge Agent stopped.")


if __name__ == "__main__":
    main()
