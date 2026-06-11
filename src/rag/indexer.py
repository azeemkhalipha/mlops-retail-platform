import os
import json
import glob
import mlflow
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
REPORTS_PATH = f"{PROJECT_ROOT}/reports"
MLFLOW_PATH  = f"file://{PROJECT_ROOT}/mlruns"
CHROMA_PATH  = f"{PROJECT_ROOT}/chroma_db"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        return client.get_collection(
            name="mlops_knowledge",
            embedding_function=embedding_fn
        )
    except Exception:
        return client.create_collection(
            name="mlops_knowledge",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )


def upsert_document(collection, doc_id, text, metadata):
    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        collection.update(ids=[doc_id], documents=[text], metadatas=[metadata])
    else:
        collection.add(ids=[doc_id], documents=[text], metadatas=[metadata])


def index_drift_reports(collection):
    files   = sorted(glob.glob(f"{REPORTS_PATH}/drift_summary_*.json"))
    indexed = 0
    for f in files:
        with open(f) as fp:
            report = json.load(fp)
        date      = report["date"]
        drift_pct = report["drift_share"] * 100
        retrain   = report["retrain_needed"]
        drifted   = report.get("drifted_features", [])

        feature_lines = []
        for feat, stats in report.get("feature_stats", {}).items():
            status = "DRIFTED" if stats["drifted"] else "stable"
            feature_lines.append(
                f"{feat}: {status} (ref_mean={stats['ref_mean']}, "
                f"curr_mean={stats['curr_mean']}, KS={stats['ks_statistic']}, "
                f"p_value={stats['p_value']})"
            )

        text = f"""Drift Report Date: {date}
Drift share: {drift_pct:.1f}% — {report['n_drifted_features']} out of {report['n_features']} features drifted
Dataset drift detected: {report['dataset_drift']}
Retraining recommended: {retrain}
Drifted features: {', '.join(drifted) if drifted else 'none'}
Feature statistics:
{chr(10).join(feature_lines)}
Conclusion: {"Retraining is needed because drift exceeds the 50% threshold." if retrain else "No retraining needed."}"""

        upsert_document(collection, f"drift_{date}", text,
            {"type": "drift_report", "date": date,
             "drift_share": report["drift_share"], "retrain_needed": str(retrain)})
        indexed += 1
    return indexed


def index_mlflow_runs(collection):
    mlflow.set_tracking_uri(MLFLOW_PATH)
    try:
        runs = mlflow.search_runs(experiment_names=["demand_forecasting"])
    except Exception:
        return 0
    if runs.empty:
        return 0

    runs = runs.sort_values("start_time", ascending=False)
    runs = runs.drop_duplicates(subset=["tags.mlflow.runName"], keep="first")

    indexed = 0
    for _, row in runs.iterrows():
        run_name = row.get("tags.mlflow.runName", "unknown")
        mae      = row.get("metrics.mae")
        rmse     = row.get("metrics.rmse")
        r2       = row.get("metrics.r2")
        run_id   = row.get("run_id", "")
        if mae is None:
            continue

        text = f"""MLflow Run: {run_name}
MAE: {mae:.4f} (average prediction error in units)
RMSE: {rmse:.4f} (penalises large errors)
R2: {r2:.4f} (explains {r2*100:.1f}% of variance)
Performance: {"BEST MODEL" if rmse < 70.0 else "Good" if rmse < 71.0 else "Average" if rmse < 73.0 else "Needs tuning"}"""

        upsert_document(collection, f"run_{run_id[:8]}", text,
            {"type": "mlflow_run", "run_name": run_name,
             "mae": float(mae), "rmse": float(rmse), "r2": float(r2)})
        indexed += 1
    return indexed


def index_project_knowledge(collection):
    docs = [
        ("knowledge_features", """Feature descriptions:
- qty_lag_1: quantity sold 1 day ago. Strongest short-term signal.
- qty_lag_7: quantity sold 7 days ago. Captures weekly seasonality.
- qty_lag_30: quantity sold 30 days ago. Captures monthly trends.
- qty_rolling_avg_7: 7-day rolling average. Smooths daily noise.
- qty_rolling_avg_30: 30-day rolling average. Shows long-term trend.
- qty_rolling_std_7: 7-day rolling standard deviation. Measures demand volatility.
- daily_revenue: total revenue that day. Combines price and volume."""),

        ("knowledge_drift", """Drift monitoring system:
- Uses Kolmogorov-Smirnov (KS) test — non-parametric, no distribution assumptions.
- p-value below 0.05 means statistically significant drift in that feature.
- KS statistic: 0 means identical distributions, 1 means completely different.
- Retrain threshold: if more than 50% of features drift, retraining is triggered.
- Reference data: first 70% of dataset (training distribution baseline).
- Current data: last 30% of dataset (simulates incoming production data).
- Currently all 7 features show drift — current data is from a later time period."""),

        ("knowledge_models", """Model comparison from MLflow (8 models trained and compared):
1. CatBoost: RMSE=69.99, MAE=21.18, R2=0.1086 — BEST MODEL
2. Linear Regression: RMSE=70.01, MAE=21.17, R2=0.1082
3. Ridge Regression: RMSE=70.01, MAE=21.17, R2=0.1082
4. Lasso Regression: RMSE=70.01, MAE=21.17, R2=0.1082
5. LightGBM: RMSE=70.95, MAE=21.29, R2=0.0841
6. Gradient Boosting: RMSE=73.33, MAE=21.38, R2=0.0215
7. Random Forest: RMSE=73.38, MAE=21.51, R2=0.0201
8. XGBoost: RMSE=74.02, MAE=21.57, R2=0.003 — needs hyperparameter tuning
CatBoost won because it handles heavy-tailed demand distributions better.
Current production model is Linear Regression v1."""),

        ("knowledge_architecture", """Platform architecture:
- PySpark processes 1M+ retail transactions into 7 lag and rolling features
- MLflow tracks 8 model experiments and registers the best model
- FastAPI serves predictions via /predict and /batch_predict endpoints
- Docker containerises the API for portable deployment
- GitHub Actions CI/CD runs tests and builds Docker on every push
- KS test drift monitor runs daily, saves HTML and JSON reports
- Airflow DAG: check_drift to decide_retrain to retrain_model or skip_retrain
- Streamlit dashboard visualises drift history and feature statistics
- RAG system answers natural language questions about model health""")
    ]

    for doc_id, text in docs:
        upsert_document(collection, doc_id, text, {"type": "project_knowledge"})
    return len(docs)


def build_index():
    print("Building RAG knowledge index...")
    collection = get_collection()
    n_drift    = index_drift_reports(collection)
    n_mlflow   = index_mlflow_runs(collection)
    n_static   = index_project_knowledge(collection)
    total      = collection.count()
    print(f"Drift reports:  {n_drift}")
    print(f"MLflow runs:    {n_mlflow}")
    print(f"Knowledge docs: {n_static}")
    print(f"Total indexed:  {total}")
    return collection


if __name__ == "__main__":
    build_index()
