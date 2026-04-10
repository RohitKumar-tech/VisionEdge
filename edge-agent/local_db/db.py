import sqlite3
import os

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Create or open SQLite DB, apply schema, return connection.

    Thread safety: each thread should call init_db() or get_conn() to get
    its own connection. SQLite WAL mode allows concurrent readers + 1 writer.
    Do NOT share a single connection across threads.
    """
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
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
