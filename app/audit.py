"""SQLite audit log. No raw text is stored: only the SHA-256 hash of the input.
Called from FastAPI BackgroundTasks so the write happens after the response is sent.
Note: on Render free tier the disk is ephemeral, so this file is lost on restart.
"""
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("AUDIT_DB_PATH", "data/audit.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_logs (
    request_id           TEXT PRIMARY KEY,
    timestamp            TEXT NOT NULL,
    input_hash           TEXT NOT NULL,
    predicted_category   TEXT NOT NULL,
    confidence           REAL NOT NULL,
    protected_prob       REAL,
    route                TEXT NOT NULL,
    decision_reason      TEXT NOT NULL,
    model_version        TEXT NOT NULL,
    latency_ms           REAL NOT NULL,
    fallback_status      TEXT NOT NULL DEFAULT 'none'
)
"""


def init_db(db_path: str = DB_PATH) -> None:
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)


def log_to_db(request_id, input_hash, predicted_category, confidence, protected_prob,
              route, decision_reason, model_version, latency_ms,
              fallback_status="none", db_path: str = DB_PATH) -> None:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO audit_logs (request_id, timestamp, input_hash, predicted_category, "
                "confidence, protected_prob, route, decision_reason, model_version, latency_ms, "
                "fallback_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, datetime.now(timezone.utc).isoformat(), input_hash,
                 predicted_category, confidence, protected_prob, route, decision_reason,
                 model_version, latency_ms, fallback_status),
            )
    except Exception as e:  # never crash the API because of the audit log
        print(f"[ERROR] Audit log failed: {e}")