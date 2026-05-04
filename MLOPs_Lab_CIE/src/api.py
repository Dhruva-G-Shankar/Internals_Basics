from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import BEST_MODEL_PATH, PREDICTION_LOG
from src.utils import save_jsonl


model = joblib.load(BEST_MODEL_PATH)

app = FastAPI(title="CopilotBench API")


class InferenceRequest(BaseModel):
    code_context_lines: int = Field(..., ge=5, le=200)
    language_complexity: int = Field(..., ge=1, le=5)
    prompt_length: int = Field(..., ge=10, le=500)
    is_inline: int = Field(..., ge=0, le=1)


@app.get("/heartbeat")
def heartbeat():
    return {
        "status": "running",
        "model": type(model).__name__,
        "version": "1.0",
    }


@app.post("/infer")
def infer(req: InferenceRequest):
    payload = req.model_dump()

    X = pd.DataFrame([payload])

    prediction = float(model.predict(X)[0])

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": payload,
        "prediction": prediction,
    }

    save_jsonl(PREDICTION_LOG, log_entry)

    return {
        "prediction": round(prediction, 6)
    }