![CI Pipeline](https://github.com/azeemkhalipha/mlops-retail-platform/actions/workflows/ci.yml/badge.svg)

# MLOps Retail Platform

End-to-end MLOps platform for retail demand forecasting. Ingests 1M+ retail transactions, engineers features with PySpark, trains and tracks 8 models with MLflow, serves predictions via a REST API, monitors for data drift daily, and automatically retrains when the model degrades.

## Architecture

![Architecture](docs/architecture.png)

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Feature engineering | PySpark | Process 1M+ rows, build lag and rolling features |
| Experiment tracking | MLflow | Compare 8 models, register best |
| Model training | scikit-learn, XGBoost, LightGBM, CatBoost | Demand forecasting |
| Model serving | FastAPI + Docker | REST API with /predict and /batch_predict |
| CI/CD | GitHub Actions | Tests and Docker build on every push |
| Drift monitoring | KS test (scipy) | Statistical drift detection across 7 features |
| Orchestration | Apache Airflow | Daily drift check with auto retraining |
| Dashboard | Streamlit + Plotly | Live monitoring with retraining trigger |

## Project structure
mlops-retail-platform/
├── src/
│   ├── features/          # PySpark feature engineering
│   ├── training/          # Retraining script with MLflow logging
│   ├── serving/           # FastAPI app, model loader, Pydantic schemas
│   ├── monitoring/        # KS test drift detector, HTML/JSON reports
│   └── dashboard/         # Streamlit monitoring dashboard
├── airflow/
│   └── dags/              # check_drift → decide → retrain/skip
├── tests/                 # pytest suite with mocked model
├── .github/workflows/     # GitHub Actions CI pipeline
├── Dockerfile
├── notebooks/             # PySpark feature exploration
└── reports/               # Generated drift reports
## Models compared

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 21.17 | 70.01 | 0.1082 |
| Ridge Regression | 21.17 | 70.01 | 0.1082 |
| Lasso Regression | 21.17 | 70.01 | 0.1082 |
| **CatBoost** | **21.18** | **69.99** | **0.1086** |
| LightGBM | 21.29 | 70.95 | 0.0841 |
| Gradient Boosting | 21.38 | 73.33 | 0.0215 |
| Random Forest | 21.51 | 73.38 | 0.0201 |
| XGBoost | 21.57 | 74.02 | 0.0031 |
| **XGBoost (Optuna tuned)** | **21.19** | **69.86** | **0.1120** |
| CatBoost (Optuna tuned) | 21.18 | 71.06 | 0.0813 |

## Running the project

**Feature engineering:**
```bash
conda activate mlops
jupyter notebook notebooks/01_feature_engineering.ipynb
```

**Drift detection:**
```bash
python src/monitoring/drift_detector.py
```

**Retrain model:**
```bash
python src/training/retrain.py
```

**Start API:**
```bash
cd src/serving
uvicorn app:app --reload --port 8000
```

**Start API with Docker:**
```bash
docker build -t retail-demand-api:v1 .
docker run -p 8001:8000 retail-demand-api:v1
```

**Run dashboard:**
```bash
streamlit run src/dashboard/app.py
```

**Run tests:**
```bash
pytest tests/ -v
```

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Model status |
| /predict | POST | Single demand prediction |
| /batch_predict | POST | Bulk predictions |
| /docs | GET | Swagger UI |

## Dataset

[Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) — 1M+ UK retail transactions, 2009–2011.

## Key decisions

- **Lag features over raw features** — lag_1, lag_7, lag_30 carry significantly more signal for time-series forecasting than raw date columns
- **KS test for drift** — non-parametric, no distribution assumptions, works on any feature type
- **Mocked model in CI** — tests pass without the pickle file in the repo, keeping the CI environment clean
- **Transparent Plotly backgrounds** — dashboard charts inherit Streamlit theme, works in light and dark mode
- **SequentialExecutor in Airflow** — lightweight for local dev; production would use KubernetesExecutor
