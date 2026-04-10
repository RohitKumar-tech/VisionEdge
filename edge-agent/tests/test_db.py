from local_db.db import init_db


def test_init_creates_tables(tmp_db_path):
    conn = init_db(tmp_db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "persons" in tables
    assert "pending_events" in tables
    assert "config_cache" in tables
    assert "sync_state" in tables


def test_insert_and_retrieve_person(tmp_db_path):
    import numpy as np
    conn = init_db(tmp_db_path)
    embedding = np.zeros(512, dtype=np.float32)
    conn.execute(
        "INSERT INTO persons (id, name, type, embedding, updated_at) VALUES (?,?,?,?,?)",
        ("p1", "Alice", "staff", embedding.tobytes(), "2026-04-07T00:00:00Z")
    )
    conn.commit()
    row = conn.execute("SELECT name, type FROM persons WHERE id=?", ("p1",)).fetchone()
    assert row[0] == "Alice"
    assert row[1] == "staff"


def test_pending_event_enqueue(tmp_db_path):
    conn = init_db(tmp_db_path)
    conn.execute(
        "INSERT INTO pending_events (id, camera_id, event_type, timestamp, created_at) VALUES (?,?,?,?,?)",
        ("e1", "cam-1", "checkin", "2026-04-07T10:00:00Z", "2026-04-07T10:00:00Z")
    )
    conn.commit()
    row = conn.execute("SELECT uploaded FROM pending_events WHERE id=?", ("e1",)).fetchone()
    assert row[0] == 0  # not yet uploaded
