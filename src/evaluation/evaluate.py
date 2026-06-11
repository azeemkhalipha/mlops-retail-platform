"""
Evaluation script — saves metrics to reports/metrics.json.
DVC tracks this file so you can compare metrics across versions
using: dvc metrics diff HEAD~1
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT  = os.getenv("PROJECT_ROOT")
FEATURES_PATH = f"{PROJECT_ROOT}/data/features"
MODEL_PATH    = f"{PROJECT_ROOT}/models/demand_forecast_v1.pkl"
METRICS_PATH  = f"{PROJECT_ROOT}/reports/metrics.json"

FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_30",
    "qty_rolling_avg_7", "qty_rolling_avg_30",
    "qty_rolling_std_7", "daily_revenue"
]

# Load data
df = pd.read_parquet(f"{FEATURES_PATH}/ml_features")
df = df.sort_values(["StockCode", "invoice_date"])
df["target"] = df.groupby("StockCode")["daily_qty"].shift(-1)
df = df.dropna(subset=["target"] + FEATURE_COLS)

X = df[FEATURE_COLS]
y = df["target"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model and evaluate
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

preds = model.predict(X_test)
mae   = float(mean_absolute_error(y_test, preds))
rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
r2    = float(r2_score(y_test, preds))

metrics = {"mae": mae, "rmse": rmse, "r2": r2}

os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved to: {METRICS_PATH}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")
