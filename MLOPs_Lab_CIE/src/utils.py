import json
import os
from math import sqrt

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ensure_dirs():
    folders = ["models", "logs", "results", "mlruns"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def load_data(path: str):
    return pd.read_csv(path)


def split_xy(df):
    X = df.drop(columns=["suggestion_accept_rate"])
    y = df["suggestion_accept_rate"]
    return X, y


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": round(float(mae), 6),
        "rmse": round(float(rmse), 6)
    }


def save_json(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def save_jsonl(path: str, payload: dict):
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")