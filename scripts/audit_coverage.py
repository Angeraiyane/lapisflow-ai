"""How many replayed requests are missing from the audit DB (e.g. after '[ERROR] Audit log failed: database is locked')?
Joins the replay trace (request_id) with data/audit.db. Run from the project root:
  python scripts/audit_coverage.py results/trace_test_20261108.csv
"""
import csv
import glob
import os
import sqlite3
import sys
from contextlib import closing

if len(sys.argv) > 1:
    trace = sys.argv[1]
else:
    candidates = glob.glob("results/trace_*.csv")
    if not candidates:
        sys.exit("No trace file found in results/. Run evaluate_live.py first.")
    trace = max(candidates, key=os.path.getmtime)
    print(f"(no argument: using the most recent trace, {trace})")
with open(trace, encoding="utf-8", newline="") as f:
    ids = [r["request_id"] for r in csv.DictReader(f) if r.get("request_id")]
with closing(sqlite3.connect(os.environ.get("AUDIT_DB_PATH", "data/audit.db"))) as conn:
    present = {r[0] for r in conn.execute("SELECT request_id FROM audit_logs")}
missing = [i for i in ids if i not in present]
print(f"requests in trace: {len(ids)} | present in audit DB: {len(ids) - len(missing)} | missing: {len(missing)} ({100 * len(missing) / max(1, len(ids)):.2f}%)")