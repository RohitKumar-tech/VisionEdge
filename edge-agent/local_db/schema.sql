-- VisionEdge Edge Agent — Local SQLite Schema
-- Stores persons (face embeddings), pending events, and config cache.

-- WAL mode: allows concurrent readers + 1 writer without "database is locked" errors.
-- Must be set before any table is created on a new DB.
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS persons (
    id          TEXT PRIMARY KEY,  -- matches cloud person.id
    name        TEXT NOT NULL,
    type        TEXT NOT NULL CHECK(type IN ('staff', 'vip', 'blacklisted')),
    embedding   BLOB NOT NULL,     -- raw float32 bytes (decrypted for local use)
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_events (
    id          TEXT PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    person_id   TEXT,              -- NULL if unknown face
    confidence  REAL,
    event_type  TEXT NOT NULL CHECK(event_type IN ('checkin','checkout','vip_spotted','blacklist_alert','unknown_face')),
    timestamp   TEXT NOT NULL,
    clip_path   TEXT,              -- local path before upload
    created_at  TEXT NOT NULL,
    uploaded    INTEGER NOT NULL DEFAULT 0 CHECK(uploaded IN (0, 1))
);

-- Index: event uploader queries WHERE uploaded=0 on every poll cycle
CREATE INDEX IF NOT EXISTS idx_pending_events_uploaded ON pending_events(uploaded);

CREATE TABLE IF NOT EXISTS config_cache (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    key         TEXT PRIMARY KEY,  -- e.g. 'last_sync_at', 'sync_cursor'
    value       TEXT NOT NULL
);

-- Attendance dedup: one row per (person, date).
-- checked_out_at is NULL until they leave; updated on every exit (tracks latest).
CREATE TABLE IF NOT EXISTS attendance_log (
    person_id      TEXT NOT NULL,
    date           TEXT NOT NULL,    -- YYYY-MM-DD (UTC)
    first_camera   TEXT NOT NULL,    -- which camera first saw them today
    checked_in_at  TEXT NOT NULL,    -- ISO timestamp of first sighting
    checked_out_at TEXT,             -- ISO timestamp of last exit today (nullable)
    PRIMARY KEY (person_id, date)
);
