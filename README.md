![CI Pipeline](https://github.com/azeemkhalipha/mlops-retail-platform/actions/workflows/ci.yml/badge.svg)

**Live API:** https://mlops-retail-platform.onrender.com/docs

# MLOps Retail Platform

End-to-end MLOps platform for retail demand forecasting. Ingests 1M+ retail transactions, engineers features with PySpark, trains and compares 8 models with MLflow, serves predictions via REST API, monitors data drift daily, auto-retrains on drift detection, explains predictions with SHAP, tunes hyperparameters with Optuna, versions data with DVC, and provides a natural language assistant powered by a local LLM.

## Architecture

![Architecture](docs/architecture.png)

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Feature engineering | PySpark | Process 1M+ rows, lag and rolling features |
| Experiment tracking | MLflow | Compare 8 models, register best |
| Model training | scikit-learn, XGBoost, LightGBM, CatBoost | Demand forecasting |
| Hyperparameter tuning | Optuna | Automated TPE search, 50 trials logged to MLflow |
| Model explainability | SHAP | Feature contribution per prediction |
| Model serving | FastAPI + Docker | /predict, /batch_predict, /explain endpoints |
| CI/CD | GitHub Actions | Tests and Docker build on every push |
| Drift monitoring | KS test (scipy) | Statistical drift detection across 7 features |
| Orchestration | Apache Airflow | Daily drift check with auto retraining |
| Data versioning | DVC | Reproduce any model run exactly |
| Dashboard | Streamlit + Plotly | Live monitoring, drift charts, prediction explainer |
| RAG assistant | Ollama + ChromaDB | Natural language queries over model data |

## Project structure
mlops-retail-platform/
├── src/
│   ├── features/          # PySpark feature engineering
│   ├── training/          # Retraining script with MLflow logging
│   ├── serving/           # FastAPI app — /predict /batch_predict /explain
│   ├── monitoring/        # KS test drift detector, HTML/JSON reports
│   ├── explainability/    # SHAP-based prediction explanations
│   ├── tuning/            # Optuna hyperparameter search
│   ├── evaluation/        # Model evaluation, DVC metrics
│   ├── rag/               # ChromaDB indexer + Ollama retriever
│   └── dashboard/         # Streamlit monitoring dashboard + floating chat
├── airflow/
│   └── dags/              # check_drift → decide → retrain/skip
├── tests/                 # pytest suite with mocked model
├── .github/workflows/     # GitHub Actions CI pipeline
├── notebooks/             # PySpark exploration + feature runner
├── reports/               # Drift reports, SHAP plots, metrics.json
├── dvc.yaml               # DVC pipeline: featurize → train → evaluate
├── Dockerfile
└── README.md
## Model comparison

All experiments tracked in MLflow. 8 baseline models + 2 Optuna-tuned models.

| Model | MAE | RMSE | R2 | Notes |
|---|---|---|---|---|
| Linear Regression | 21.17 | 70.01 | 0.1082 | Strong baseline |
| Ridge Regression | 21.17 | 70.01 | 0.1082 | L2 regularisation |
| Lasso Regression | 21.17 | 70.01 | 0.1082 | L1 — feature selection |
| **CatBoost** | **21.18** | **69.99** | **0.1086** | Best untuned model |
| LightGBM | 21.29 | 70.95 | 0.0841 | Fast gradient boosting |
| Gradient Boosting | 21.38 | 73.33 | 0.0215 | sklearn ensemble |
| Random Forest | 21.51 | 73.38 | 0.0201 | Bagging ensemble |
| XGBoost | 21.57 | 74.02 | 0.0031 | Underperforms without tuning |
| **XGBoost (Optuna tuned)** | **21.19** | **69.86** | **0.1120** | Best overall — 5.6% RMSE improvement |
| CatBoost (Optuna tuned) | 21.18 | 71.06 | 0.0813 | Defaults were near-optimal |

**Key finding:** Optuna tuning improved XGBoost RMSE from 74.02 to 69.86 — a 5.6% improvement in 30 automated trials. Optimal params: `max_depth=3`, `learning_rate=0.013`, `n_estimators=345`.

## Current model metrics (DVC tracked)

```bash
dvc metrics show
```
Path                  mae       r2       rmse
reports/metrics.json  21.168    0.108    70.006
Compare metrics across versions:
```bash
dvc metrics diff HEAD~1
```

## SHAP explainability

Every prediction can be explained. Example output:
Predicted demand: 13.9 units
Base value: 23.87 units
Feature contributions:
▼ qty_rolling_avg_30  : -9.88  (high impact)
▼ qty_rolling_avg_7   : -6.70  (high impact)
▲ qty_rolling_std_7   : +5.95  (high impact)
▲ qty_lag_30          : +0.38  (low impact
SHAP plots saved to `reports/shap/`.

## RAG assistant

A floating chat widget in the Streamlit dashboard powered by Llama 3.2 (local, free).

- Drift reports and MLflow metadata indexed into ChromaDB
- sentence-transformers for local embeddings
- Ollama serves Llama 3.2 locally — no API key, no cost
- Context-scoped: only answers questions about this platform

Example questions:
- "Which model performed best?"
- "Should I retrain right now?"
- "Which features drifted the most?"

## Running the project

**Feature engineering:**
```bash
conda activate mlops
python notebooks/run_features.py
```

**Train all models:**
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

**Optuna hyperparameter tuning:**
```bash
python src/tuning/tune.py
```

**Drift detection:**
```bash
python src/monitoring/drift_detector.py
```

**Retrain model:**
```bash
python src/training/retrain.py
```

**Evaluate and track metrics:**
```bash
python src/evaluation/evaluate.py
dvc metrics show
```

**Explain a prediction:**
```bash
python src/explainability/explainer.py
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

**Start dashboard:**
```bash
OLLAMA_ORIGINS='*' ollama serve &
streamlit run src/dashboard/app.py
```

**Run tests:**
```bash
pytest tests/ -v
```

**Run full DVC pipeline:**
```bash
dvc repro
```

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Model and API status |
| /predict | POST | Single demand prediction |
| /batch_predict | POST | Bulk predictions |
| /explain | POST | SHAP explanation for a prediction |
| /docs | GET | Interactive Swagger UI |

## Dataset

[Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) — 1M+ UK retail transactions, 2009–2011.

## Key engineering decisions

- **Lag features over raw features** — qty_lag_1, qty_lag_7, qty_lag_30 carry significantly more forecasting signal than raw date columns
- **KS test for drift** — non-parametric, no distribution assumptions, works on any feature type
- **Optuna TPE over random search** — learns from previous trials, converges faster to optimal hyperparameters
- **SHAP KernelExplainer** — model-agnostic, works with any sklearn-compatible model
- **Local RAG with Ollama** — zero cost, no API key, runs on Apple Silicon, context-scoped to prevent hallucination
- **Mocked model in CI** — tests pass without the pickle file in the repo
- **DVC for reproducibility** — any model run can be reproduced exactly with `git checkout <commit> && dvc pull`
- **Transparent Plotly backgrounds** — dashboard charts inherit Streamlit theme, works in light and dark mode
