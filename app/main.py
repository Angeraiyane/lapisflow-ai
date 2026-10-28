import json
import time
import uuid
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

# Constants
MODEL_PATH = "models/clf_v1.joblib"
THRESHOLDS_PATH = "data/thresholds.json"
PROTECTED_CATEGORY = "unauthorized_transactions_security"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and threshold configuration into app state
    try:
        app.state.model = joblib.load(MODEL_PATH)
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            app.state.thresholds = json.load(f)

        classes = list(app.state.model.classes_)
        if PROTECTED_CATEGORY in classes:
            app.state.protected_category_index = classes.index(PROTECTED_CATEGORY)
        else:
            raise ValueError(f"Protected category '{PROTECTED_CATEGORY}' not found in model classes.")

        print(
            f"[INFO] Artifacts loaded successfully. "
            f"tau_protected={app.state.thresholds['tau_protected']}, "
            f"tau_confidence={app.state.thresholds['tau_confidence']}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {e}")
        app.state.model = None
        app.state.thresholds = None
        app.state.protected_category_index = None

    yield

    # Shutdown: Clean up resources if necessary
    print("[INFO] Application shutting down.")


app = FastAPI(
    title="Lapisflow AI - Cascade Triage",
    description="Customer-ticket triage engine with confidence gating, LLM fallback, and protected intent escalation.",
    version="0.2.0",
    lifespan=lifespan
)


# Pydantic Models
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
    is_ready = model_ready and thresholds_ready

    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "lapisflow-ai",
        "model_loaded": model_ready,
        "thresholds_loaded": thresholds_ready,
        "version": "0.2.0"
    }


@app.post("/triage", response_model=TriageResponse)
def triage_ticket(payload: TriageRequest, request: Request):
    model = getattr(request.app.state, "model", None)
    thresholds = getattr(request.app.state, "thresholds", None)
    protected_idx = getattr(request.app.state, "protected_category_index", None)

    if model is None or thresholds is None or protected_idx is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model or thresholds not loaded."
        )

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Inférence probabiliste
    probs = model.predict_proba([payload.text])[0]
    classes = list(model.classes_)
    top_idx = int(np.argmax(probs))
    top_category = classes[top_idx]
    confidence = float(probs[top_idx])
    protected_prob = float(probs[protected_idx])

    tau_p = thresholds["tau_protected"]
    tau_c = thresholds["tau_confidence"]

    # Cascade Decision Gates
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
        "model_version": "clf_v1",
        "latency_ms": latency_ms
    }

    # Structured stdout JSON log line
    log_record = {
        "timestamp": time.time(),
        "request_id": request_id,
        "input_hash": hashlib.sha256(payload.text.encode("utf-8")).hexdigest(),
        "decision": response_payload
    }
    print(json.dumps(log_record))

    return response_payload


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)