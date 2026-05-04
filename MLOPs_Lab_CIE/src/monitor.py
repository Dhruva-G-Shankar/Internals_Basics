import json

import pandas as pd

from src.config import (
    TRAIN_PATH,
    PREDICTION_LOG,
    CODE_CONTEXT_SHIFT_THRESHOLD,
    PROMPT_LENGTH_SHIFT_THRESHOLD,
)
from src.utils import load_data, save_json


def main():
    train_df = load_data(TRAIN_PATH)

    train_code_mean = train_df["code_context_lines"].mean()
    train_prompt_mean = train_df["prompt_length"].mean()

    logs = []

    with open(PREDICTION_LOG, "r") as f:
        for line in f:
            logs.append(json.loads(line))

    live_inputs = [x["input"] for x in logs]
    live_df = pd.DataFrame(live_inputs)

    live_code_mean = live_df["code_context_lines"].mean()
    live_prompt_mean = live_df["prompt_length"].mean()

    alerts = []

    code_shift = abs(live_code_mean - train_code_mean)
    code_status = "ALERT" if code_shift > CODE_CONTEXT_SHIFT_THRESHOLD else "OK"

    alerts.append({
        "feature": "code_context_lines",
        "train_mean": round(float(train_code_mean), 2),
        "live_mean": round(float(live_code_mean), 2),
        "shift": round(float(code_shift), 2),
        "threshold": CODE_CONTEXT_SHIFT_THRESHOLD,
        "status": code_status,
    })

    prompt_shift = abs(live_prompt_mean - train_prompt_mean)
    prompt_status = "ALERT" if prompt_shift > PROMPT_LENGTH_SHIFT_THRESHOLD else "OK"

    alerts.append({
        "feature": "prompt_length",
        "train_mean": round(float(train_prompt_mean), 2),
        "live_mean": round(float(live_prompt_mean), 2),
        "shift": round(float(prompt_shift), 2),
        "threshold": PROMPT_LENGTH_SHIFT_THRESHOLD,
        "status": prompt_status,
    })

    preds = [x["prediction"] for x in logs]

    payload = {
        "total_predictions": len(logs),
        "mean_prediction": round(sum(preds) / len(preds), 6),
        "drift_detected": any(a["status"] == "ALERT" for a in alerts),
        "alerts": alerts,
    }

    save_json("results/step4_s5.json", payload)

    print(payload)


if __name__ == "__main__":
    main()