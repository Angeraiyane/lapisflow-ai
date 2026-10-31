"""Invariant and boundary tests for /triage. Run from the project root:  python -m pytest -v

Uses a SEPARATE audit database (tests/_audit_test.db) so test rows never mix with real audit data.
"""
import os
import sqlite3
import time
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from app.main import PROTECTED_CATEGORY, app

DB = "tests/_audit_test.db"  # same path as tests/conftest.py


class StubModel:
    """Returns fixed probabilities so we can test exact threshold boundaries."""

    def __init__(self, classes, probs):
        self.classes_ = np.array(classes)
        self._probs = np.array(probs, dtype=float)

    def predict_proba(self, X):
        return np.array([self._probs for _ in X])


def _stub_probs(client, top_class, top_p, protected_p):
    classes = list(app.state.model.classes_)
    probs = np.zeros(len(classes))
    probs[classes.index(top_class)] = top_p
    pi = classes.index(PROTECTED_CATEGORY)
    if top_class != PROTECTED_CATEGORY:
        probs[pi] = protected_p
    return classes, probs


def _post_with_stub(client, top_class, top_p, protected_p, text="boundary test message"):
    classes, probs = _stub_probs(client, top_class, top_p, protected_p)
    real = app.state.model
    app.state.model = StubModel(classes, probs)
    try:
        r = client.post("/triage", json={"text": text})
    finally:
        app.state.model = real
    assert r.status_code == 200
    return r.json()


def _route_with_stub(client, top_class, top_p, protected_p, text="boundary test message"):
    return _post_with_stub(client, top_class, top_p, protected_p, text)["route"]


def _non_protected_class():
    return next(c for c in app.state.model.classes_ if c != PROTECTED_CATEGORY)


def test_health_is_healthy(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_protected_probability_boundary(client):
    tau_p = app.state.thresholds["tau_protected"]
    other = _non_protected_class()
    # exactly at tau_protected -> human, even with very high confidence
    assert _route_with_stub(client, other, 0.99, tau_p) == "human"
    # just below tau_protected and above tau_confidence -> auto
    assert _route_with_stub(client, other, 0.99, tau_p - 1e-6) == "auto"


def test_confidence_boundary(client):
    tau_c = app.state.thresholds["tau_confidence"]
    other = _non_protected_class()
    assert _route_with_stub(client, other, tau_c, 0.0) == "auto"          # exactly at tau
    assert _route_with_stub(client, other, tau_c - 0.001, 0.0) == "human"  # just below


def test_protected_top_class_is_never_auto(client):
    assert _route_with_stub(client, PROTECTED_CATEGORY, 0.999, 0.999) == "human"


def test_no_validation_protected_message_is_auto(client):
    """The invariant behind the leakage numbers: 0 leaks OBSERVED on the validation set."""
    if not os.path.exists("data/val.csv"):
        pytest.skip("data/val.csv not available")
    val = pd.read_csv("data/val.csv")
    prot = val[val["is_protected"] == 1]
    leaks = [t for t in prot["text"] if client.post("/triage", json={"text": t}).json()["route"] != "human"]
    assert len(leaks) == 0, f"{len(leaks)} protected validation messages were not routed to human"


def test_audit_row_has_hash_and_no_raw_text(client):
    text = "UNIQUE-PRIVACY-CANARY-12345 my card is somewhere"
    r = client.post("/triage", json={"text": text})
    rid = r.json()["request_id"]
    time.sleep(0.3)
    with closing(sqlite3.connect(DB)) as conn:
        row = conn.execute("SELECT * FROM audit_logs WHERE request_id = ?", (rid,)).fetchone()
        cols = [d[0] for d in conn.execute("SELECT * FROM audit_logs").description]
        dump = " ".join(str(v) for r2 in conn.execute("SELECT * FROM audit_logs") for v in r2)
    assert row is not None
    assert "input_hash" in cols and not any("text" == c for c in cols)
    assert "UNIQUE-PRIVACY-CANARY-12345" not in dump


def test_audit_row_records_thresholds_and_guard_version(client):
    rid = client.post("/triage", json={"text": "How do I top up my account?"}).json()["request_id"]
    time.sleep(0.3)
    with closing(sqlite3.connect(DB)) as conn:
        tp, tc, gv = conn.execute("SELECT tau_protected, tau_confidence, guard_version FROM audit_logs WHERE request_id=?", (rid,)).fetchone()
    assert tp == app.state.thresholds["tau_protected"] and tc == app.state.thresholds["tau_confidence"]
    assert gv.startswith("lex_")


def test_same_input_same_hash(client):
    ids = [client.post("/triage", json={"text": "hash stability check"}).json()["request_id"] for _ in range(2)]
    time.sleep(0.3)
    with closing(sqlite3.connect(DB)) as conn:
        hashes = {conn.execute("SELECT input_hash FROM audit_logs WHERE request_id=?", (i,)).fetchone()[0] for i in ids}
    assert len(hashes) == 1


def test_too_short_text_rejected(client):
    assert client.post("/triage", json={"text": "hi"}).status_code == 422


def test_lexical_gate_forces_human_even_when_model_is_confident_and_safe(client):
    other = _non_protected_class()
    for text in ["I lost my card", "can you locate my card", "guide me to my card's location"]:
        out = _post_with_stub(client, other, 0.999, 0.0, text)
        assert out["route"] == "human", text
        assert "Lexical" in out["reason"], text


def test_lexical_gate_does_not_fire_on_plain_message(client):
    other = _non_protected_class()
    assert _route_with_stub(client, other, 0.999, 0.0, "How do I top up my account?") == "auto"


def test_audit_records_decision_layer(client):
    other = _non_protected_class()
    rid = _post_with_stub(client, other, 0.999, 0.0, "I lost my card")["request_id"]
    rid2 = _post_with_stub(client, other, 0.999, 0.0, "How do I top up my account?")["request_id"]
    time.sleep(0.3)
    with closing(sqlite3.connect(DB)) as conn:
        layer1 = conn.execute("SELECT decision_layer FROM audit_logs WHERE request_id=?", (rid,)).fetchone()[0]
        layer2 = conn.execute("SELECT decision_layer FROM audit_logs WHERE request_id=?", (rid2,)).fetchone()[0]
    assert layer1 == "lexical_gate" and layer2 == "confidence_gate"


def test_old_audit_db_is_migrated(tmp_path):
    from app.audit import init_db
    db = str(tmp_path / "old.db")
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE audit_logs (request_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, "
                     "input_hash TEXT NOT NULL, predicted_category TEXT NOT NULL, confidence REAL NOT NULL, "
                     "protected_prob REAL, route TEXT NOT NULL, decision_reason TEXT NOT NULL, "
                     "model_version TEXT NOT NULL, latency_ms REAL NOT NULL, fallback_status TEXT NOT NULL DEFAULT 'none')")
    init_db(db)
    with closing(sqlite3.connect(db)) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(audit_logs)")]
        for c in ("decision_layer", "tau_protected", "tau_confidence", "guard_version"):
            assert c in cols


def test_stats_counts_requests_by_route(client):
    for text in ["How do I top up my account?", "I lost my card"]:
        client.post("/triage", json={"text": text})
    time.sleep(0.3)
    s = client.get("/stats").json()
    assert s["total_requests"] >= 2
    assert sum(s["by_route"].values()) == s["total_requests"]
    assert sum(s["by_decision_layer"].values()) == s["total_requests"]