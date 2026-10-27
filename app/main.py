from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="Lapisflow AI - Cascade Triage",
    description="Customer-ticket triage engine with confidence gating, LLM fallback, and protected intent escalation.",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "lapisflow-ai",
        "model_loaded": True,
        "version": "0.1.0"
    }

if __name__ == "__main__":
    # Avant uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)