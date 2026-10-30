import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.audit import init_db, log_to_db

MODEL_PATH = "models/clf_v1.joblib"
THRESHOLDS_PATH = "data/thresholds.json"
PROTECTED_CATEGORY = "unauthorized_transactions_security"
MODEL_VERSION = "clf_v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = joblib.load(MODEL_PATH)
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            app.state.thresholds = json.load(f)

        classes = list(app.state.model.classes_)
        if PROTECTED_CATEGORY not in classes:
            raise ValueError(f"Protected category '{PROTECTED_CATEGORY}' not found in model classes.")
        app.state.protected_category_index = classes.index(PROTECTED_CATEGORY)

        init_db()
        print(f"[INFO] Artifacts loaded. tau_protected={app.state.thresholds['tau_protected']}, "
              f"tau_confidence={app.state.thresholds['tau_confidence']}")
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {e}")
        app.state.model = None
        app.state.thresholds = None
        app.state.protected_category_index = None
    yield
    print("[INFO] Application shutting down.")


app = FastAPI(
    title="Lapisflow AI - Cascade Triage",
    description="Customer-ticket triage with a protected-intent gate, a confidence gate and an audit log.",
    version="0.3.0",
    lifespan=lifespan,
)


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=3, examples=["My credit card was stolen yesterday."])


class TriageResponse(BaseModel):
    request_id: str
    route: str  # "auto" | "human"
    category: str
    confidence: float
    reason: str
    model_version: str
    latency_ms: float


@app.get("/health")
def health_check():
    model_ready = getattr(app.state, "model", None) is not None
    thresholds_ready = getattr(app.state, "thresholds", None) is not None
    return {
        "status": "healthy" if (model_ready and thresholds_ready) else "degraded",
        "service": "lapisflow-ai",
        "model_loaded": model_ready,
        "thresholds_loaded": thresholds_ready,
        "version": "0.3.0",
    }


@app.post("/triage", response_model=TriageResponse)
def triage_ticket(payload: TriageRequest, background_tasks: BackgroundTasks, request: Request):
    model = getattr(request.app.state, "model", None)
    thresholds = getattr(request.app.state, "thresholds", None)
    protected_idx = getattr(request.app.state, "protected_category_index", None)
    if model is None or thresholds is None or protected_idx is None:
        raise HTTPException(status_code=503, detail="Service unavailable: model or thresholds not loaded.")

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    input_hash = hashlib.sha256(payload.text.encode("utf-8")).hexdigest()

    probs = model.predict_proba([payload.text])[0]
    classes = list(model.classes_)
    top_idx = int(np.argmax(probs))
    top_category = classes[top_idx]
    confidence = float(probs[top_idx])
    protected_prob = float(probs[protected_idx])

    tau_p = thresholds["tau_protected"]
    tau_c = thresholds["tau_confidence"]

    if protected_prob >= tau_p:
        route = "human"
        reason = f"Safety policy triggered: protected probability ({protected_prob:.4f}) >= tau_protected ({tau_p:.4f})"
    elif confidence >= tau_c:
        route = "auto"
        reason = f"Confidence threshold satisfied ({confidence:.4f} >= tau_confidence ({tau_c:.4f}))"
    else:
        route = "human"
        reason = f"Low confidence escalation ({confidence:.4f} < tau_confidence ({tau_c:.4f}))"

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response_payload = {
        "request_id": request_id,
        "route": route,
        "category": top_category,
        "confidence": round(confidence, 4),
        "reason": reason,
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
    }

    # Audit write happens AFTER the response is sent (not on the hot path).
    background_tasks.add_task(
        log_to_db,
        request_id=request_id,
        input_hash=input_hash,
        predicted_category=top_category,
        confidence=confidence,
        protected_prob=protected_prob,
        route=route,
        decision_reason=reason,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )

    print(json.dumps({"timestamp": time.time(), "request_id": request_id,
                      "input_hash": input_hash, "decision": response_payload}))
    return response_payload


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)