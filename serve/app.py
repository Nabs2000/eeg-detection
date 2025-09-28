import json, os, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis

# Config via env vars
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MODEL_DIR  = Path(os.getenv("MODEL_DIR", "/app/model"))
MODEL_PATH = MODEL_DIR / "model.pt"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

app = FastAPI(title="Docker Day API", version="0.1.0")
r = None

class PredictRequest(BaseModel):
    values: list[float]

@app.on_event("startup")
def startup():
    global r
    # load artifacts
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise RuntimeError("Model artifacts missing")
    # “Load” model (mock: read text file)
    _ = MODEL_PATH.read_text()
    _j = json.loads(THRESHOLD_PATH.read_text())

    # connect redis
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
    except Exception as e:
        raise RuntimeError(f"Redis not reachable: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/metrics")
def metrics():
    c = r.get("predict_calls") or 0
    return {"predict_calls": int(c)}

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.values:
        raise HTTPException(400, "values must be non-empty")
    # mock “probability”: bounded average
    prob = max(0.0, min(1.0, sum(req.values) / (10.0 * len(req.values))))
    r.incr("predict_calls")
    return {"prob": prob, "label": int(prob >= 0.5)}
