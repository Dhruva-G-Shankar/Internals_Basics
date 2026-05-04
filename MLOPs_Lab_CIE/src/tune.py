import itertools
import random
import joblib
import mlflow
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_PATH,
    BEST_MODEL_PATH,
)
from src.utils import (
    load_data,
    split_xy,
    compute_metrics,
    save_json,
)


def main():
    mlflow.set_tracking_uri("file:./mlruns")

    df = load_data(TRAIN_PATH)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }

    combinations = list(
        itertools.product(
            param_grid["n_estimators"],
            param_grid["max_depth"],
            param_grid["min_samples_split"],
        )
    )

    random.seed(RANDOM_STATE)
    random.shuffle(combinations)

    best_model = None
    best_params = None
    best_rmse = float("inf")
    best_mae = None
    best_cv_mae = None

    mlflow.set_experiment("copilotbench-suggestion-accept-rate")

    with mlflow.start_run(run_name="tuning-copilotbench"):
        for i, combo in enumerate(combinations, start=1):
            params = {
                "n_estimators": combo[0],
                "max_depth": combo[1],
                "min_samples_split": combo[2],
            }

            with mlflow.start_run(
                run_name=f"trial_{i}",
                nested=True
            ):
                model = RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    **params
                )

                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="neg_mean_absolute_error",
                )

                cv_mae = abs(np.mean(cv_scores))

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                metrics = compute_metrics(y_test, preds)

                for k, v in params.items():
                    mlflow.log_param(k, v)

                mlflow.log_metric("mae", metrics["mae"])
                mlflow.log_metric("rmse", metrics["rmse"])
                mlflow.log_metric("cv_mae", cv_mae)

                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_model = model
                    best_params = params
                    best_mae = metrics["mae"]
                    best_cv_mae = cv_mae

    joblib.dump(best_model, BEST_MODEL_PATH)

    payload = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": len(combinations),
        "best_params": best_params,
        "best_mae": round(float(best_mae), 6),
        "best_cv_mae": round(float(best_cv_mae), 6),
        "parent_run_name": "tuning-copilotbench",
    }

    save_json("results/step2_s2.json", payload)

    print("Task 2 complete")
    print(payload)


if __name__ == "__main__":
    main()