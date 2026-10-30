"""Shared test setup: ONE client for the whole session and a separate audit DB."""
import os

os.environ["AUDIT_DB_PATH"] = "tests/_audit_test.db"  # must be set BEFORE importing app

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    db = os.environ["AUDIT_DB_PATH"]
    for path in (db, db + "-wal", db + "-shm", db + "-journal"):
        try:
            os.remove(path)
        except (FileNotFoundError, PermissionError):
            pass  # Windows may hold a leftover handle; tests do not need an empty DB
    with TestClient(app) as c:
        yield c