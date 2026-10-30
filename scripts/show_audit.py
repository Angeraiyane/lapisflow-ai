"""Show audit rows (for your proof screenshot). No raw text exists in this table.
Usage:  python scripts\show_audit.py            -> last 5 rows
        python scripts\show_audit.py ID1 ID2    -> rows with those request_ids
"""
import sqlite3
import sys

with sqlite3.connect("data/audit.db") as conn:
    total, first, last = conn.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM audit_logs").fetchone()
    print(f"rows={total} first={first} last={last}\n")
    cols = "request_id, timestamp, input_hash, predicted_category, confidence, protected_prob, route, latency_ms, fallback_status"
    if len(sys.argv) > 1:
        q = f"SELECT {cols} FROM audit_logs WHERE request_id IN ({','.join('?' * (len(sys.argv) - 1))})"
        rows = conn.execute(q, sys.argv[1:]).fetchall()
    else:
        rows = conn.execute(f"SELECT {cols} FROM audit_logs ORDER BY timestamp DESC LIMIT 5").fetchall()
    for r in rows:
        print(r)