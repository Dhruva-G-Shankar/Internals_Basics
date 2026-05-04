import os
import joblib
import mlflow

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    EXPERIMENT_NAME,
    TRAIN_PATH,
    BEST_MODEL_PATH,
)
from src.utils import (
    ensure_dirs,
    load_data,
    split_xy,
    compute_metrics,
    save_json,
)


def log_model_run(model_name, model, metrics):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("project_phase", "model_selection")

        for k, v in model.get_params().items():
            mlflow.log_param(k, v)

        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("rmse", metrics["rmse"])


def main():
    ensure_dirs()
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(TRAIN_PATH)

    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
    }

    results = []
    best_model = None
    best_model_name = None
    best_rmse = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        metrics = compute_metrics(y_test, preds)

        log_model_run(name, model, metrics)

        model_path = f"models/{name}.joblib"
        joblib.dump(model, model_path)

        results.append(
            {
                "name": name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model = model
            best_model_name = name

    joblib.dump(best_model, BEST_MODEL_PATH)

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results,
        "best_model": best_model_name,
        "best_metric_name": "rmse",
        "best_metric_value": round(float(best_rmse), 6),
    }

    save_json("results/step1_s1.json", payload)

    print("Task 1 complete")
    print(payload)


if __name__ == "__main__":
    main()