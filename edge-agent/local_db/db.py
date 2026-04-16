import sqlite3
import os

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

# Columns to add to existing tables when upgrading an older DB.
# Format: (table, column, definition)
_MIGRATIONS = [
    ("attendance_log", "checked_out_at", "TEXT"),  # v2: checkout support
]


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Create or open SQLite DB, apply schema, run column migrations, return connection.

    Thread safety: each thread should call init_db() or get_conn() to get
    its own connection. SQLite WAL mode allows concurrent readers + 1 writer.
    Do NOT share a single connection across threads.
    """
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    # Add any missing columns to existing tables (SQLite has no IF NOT EXISTS for ALTER)
    for table, column, definition in _MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    conn.commit()
    return conn


def get_conn(db_path: str) -> sqlite3.Connection:
    """
    Open existing DB for the calling thread.
    Caller is responsible for closing when done.
    """
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    return conn
