"""Invariant and boundary tests for /triage. Run from the project root:  python -m pytest -v

Uses a SEPARATE audit database (tests/_audit_test.db) so test rows never mix with real audit data.
"""
import os
import sqlite3
import time

os.environ["AUDIT_DB_PATH"] = "tests/_audit_test.db"  # must be set BEFORE importing app

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import PROTECTED_CATEGORY, app

DB = os.environ["AUDIT_DB_PATH"]


class StubModel:
    """Returns fixed probabilities so we can test exact threshold boundaries."""

    def __init__(self, classes, probs):
        self.classes_ = np.array(classes)
        self._probs = np.array(probs, dtype=float)

    def predict_proba(self, X):
        return np.array([self._probs for _ in X])


@pytest.fixture(scope="module")
def client():
    if os.path.exists(DB):
        os.remove(DB)
    with TestClient(app) as c:
        yield c


def _stub_probs(client, top_class, top_p, protected_p):
    classes = list(app.state.model.classes_)
    probs = np.zeros(len(classes))
    probs[classes.index(top_class)] = top_p
    pi = classes.index(PROTECTED_CATEGORY)
    if top_class != PROTECTED_CATEGORY:
        probs[pi] = protected_p
    return classes, probs


def _route_with_stub(client, top_class, top_p, protected_p):
    classes, probs = _stub_probs(client, top_class, top_p, protected_p)
    real = app.state.model
    app.state.model = StubModel(classes, probs)
    try:
        r = client.post("/triage", json={"text": "boundary test message"})
    finally:
        app.state.model = real
    assert r.status_code == 200
    return r.json()["route"]


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
    with sqlite3.connect(DB) as conn:
        row = conn.execute("SELECT * FROM audit_logs WHERE request_id = ?", (rid,)).fetchone()
        cols = [d[0] for d in conn.execute("SELECT * FROM audit_logs").description]
        dump = " ".join(str(v) for r2 in conn.execute("SELECT * FROM audit_logs") for v in r2)
    assert row is not None
    assert "input_hash" in cols and not any("text" == c for c in cols)
    assert "UNIQUE-PRIVACY-CANARY-12345" not in dump


def test_same_input_same_hash(client):
    ids = [client.post("/triage", json={"text": "hash stability check"}).json()["request_id"] for _ in range(2)]
    time.sleep(0.3)
    with sqlite3.connect(DB) as conn:
        hashes = {conn.execute("SELECT input_hash FROM audit_logs WHERE request_id=?", (i,)).fetchone()[0] for i in ids}
    assert len(hashes) == 1


def test_too_short_text_rejected(client):
    assert client.post("/triage", json={"text": "hi"}).status_code == 422