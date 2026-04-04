-- VisionEdge Edge Agent — Local SQLite Schema
-- Stores persons (face embeddings), pending events, and config cache.

CREATE TABLE IF NOT EXISTS persons (
    id          TEXT PRIMARY KEY,  -- matches cloud person.id
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,     -- 'staff' | 'vip' | 'blacklisted'
    embedding   BLOB NOT NULL,     -- raw float32 bytes (decrypted for local use)
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_events (
    id          TEXT PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    person_id   TEXT,              -- NULL if unknown face
    confidence  REAL,
    event_type  TEXT NOT NULL,     -- 'checkin' | 'checkout' | 'vip_spotted' | 'blacklist_alert' | 'unknown_face'
    timestamp   TEXT NOT NULL,
    clip_path   TEXT,              -- local path before upload
    created_at  TEXT NOT NULL,
    uploaded    INTEGER DEFAULT 0  -- 0 = pending, 1 = uploaded
);

CREATE TABLE IF NOT EXISTS config_cache (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    key         TEXT PRIMARY KEY,  -- e.g. 'last_sync_at', 'sync_cursor'
    value       TEXT NOT NULL
);
