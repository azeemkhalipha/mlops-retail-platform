import os
import pickle
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT  = os.getenv("PROJECT_ROOT")
FEATURES_PATH = f"{PROJECT_ROOT}/data/features"
MLFLOW_PATH   = f"file://{PROJECT_ROOT}/mlruns"
MODEL_PATH    = f"{PROJECT_ROOT}/models/demand_forecast_v1.pkl"

FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_30",
    "qty_rolling_avg_7", "qty_rolling_avg_30",
    "qty_rolling_std_7", "daily_revenue"
]


def load_features() -> pd.DataFrame:
    """Load ML features and create target variable."""
    df = pd.read_parquet(f"{FEATURES_PATH}/ml_features")
    df = df.sort_values(["StockCode", "invoice_date"])

    # Target = next day's quantity
    df["target"] = df.groupby("StockCode")["daily_qty"].shift(-1)
    df = df.dropna(subset=["target"] + FEATURE_COLS)

    return df


def train_model(df: pd.DataFrame) -> dict:
    """
    Train a new model, log to MLflow, save pickle.
    Returns metrics dict.
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(MLFLOW_PATH)
    mlflow.set_experiment("demand_forecasting")

    # Name the run with timestamp so you can track when retraining happened
    run_name = f"retrain_{datetime.today().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae   = float(mean_absolute_error(y_test, preds))
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2    = float(r2_score(y_test, preds))

        # Log what triggered this retraining
        mlflow.log_param("trigger",    "drift_detected")
        mlflow.log_param("train_size", int(len(X_train)))
        mlflow.log_param("test_size",  int(len(X_test)))
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"Retrain complete:")
        print(f"  Run name: {run_name}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  R2:       {r2:.4f}")

    # Save updated model to disk for the API to use
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved: {MODEL_PATH}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    print("Starting retraining pipeline...")
    df      = load_features()
    print(f"Loaded {len(df):,} rows")
    metrics = train_model(df)
    print(f"Done: {metrics}")
