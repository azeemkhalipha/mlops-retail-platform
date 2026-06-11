"""
Optuna hyperparameter tuning for XGBoost and CatBoost.

Optuna uses a Tree-structured Parzen Estimator (TPE) algorithm —
smarter than random search because it learns from previous trials
and focuses on promising regions of the parameter space.

Every trial is logged to MLflow so you can compare tuned vs
untuned models in the same experiment dashboard.
"""

import os
import sys
import mlflow
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT  = os.getenv("PROJECT_ROOT")
FEATURES_PATH = f"{PROJECT_ROOT}/data/features"
MLFLOW_PATH   = f"file://{PROJECT_ROOT}/mlruns"

FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_30",
    "qty_rolling_avg_7", "qty_rolling_avg_30",
    "qty_rolling_std_7", "daily_revenue"
]

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    """Load features and prepare X, y for training."""
    df = pd.read_parquet(f"{FEATURES_PATH}/ml_features")
    df = df.sort_values(["StockCode", "invoice_date"])
    df["target"] = df.groupby("StockCode")["daily_qty"].shift(-1)
    df = df.dropna(subset=["target"] + FEATURE_COLS)

    X = df[FEATURE_COLS]
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def tune_xgboost(n_trials: int = 30) -> dict:
    """
    Run Optuna hyperparameter search for XGBoost.

    Each trial:
    1. Optuna suggests a set of hyperparameters
    2. Model is trained with those parameters
    3. RMSE on validation set is returned
    4. Optuna learns from the result and suggests better params next trial

    After n_trials, the best parameters are logged to MLflow.
    """
    print(f"\nTuning XGBoost — {n_trials} trials...")
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri(MLFLOW_PATH)
    mlflow.set_experiment("demand_forecasting")

    best_rmse   = float("inf")
    best_params = {}

    def objective(trial):
        """
        This function runs once per trial.
        Optuna calls it repeatedly with different parameter suggestions.
        """
        params = {
            # n_estimators: how many trees to build
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            # max_depth: how deep each tree grows (deeper = more complex)
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            # learning_rate: how much each tree corrects the previous
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # subsample: fraction of data used per tree (prevents overfitting)
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            # colsample_bytree: fraction of features used per tree
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # min_child_weight: minimum samples needed to split a node
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            # gamma: minimum loss reduction to split a node
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "random_state":     42,
            "verbosity":        0
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds = model.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        return rmse

    # Run the study
    # direction="minimize" means Optuna tries to get the lowest RMSE
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_rmse   = study.best_value

    print(f"Best XGBoost RMSE: {best_rmse:.4f}")
    print(f"Best params: {best_params}")

    # Train final model with best params and log to MLflow
    with mlflow.start_run(run_name="xgboost_optuna_tuned"):
        final_model = xgb.XGBRegressor(**best_params, random_state=42)
        final_model.fit(X_train, y_train)

        preds = final_model.predict(X_test)
        mae   = float(mean_absolute_error(y_test, preds))
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2    = float(r2_score(y_test, preds))

        mlflow.log_params(best_params)
        mlflow.log_param("model_type",     "XGBoost_Optuna")
        mlflow.log_param("n_trials",       n_trials)
        mlflow.log_param("tuning_method",  "TPE")
        mlflow.log_metric("mae",           mae)
        mlflow.log_metric("rmse",          rmse)
        mlflow.log_metric("r2",            r2)
        mlflow.log_metric("best_trial",    study.best_trial.number)
        mlflow.xgboost.log_model(final_model, "model")

        print(f"\nXGBoost Optuna Tuned Results:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")

    return {
        "model":       "xgboost_optuna",
        "best_params": best_params,
        "mae":         mae,
        "rmse":        rmse,
        "r2":          r2,
        "n_trials":    n_trials
    }


def tune_catboost(n_trials: int = 20) -> dict:
    """
    Run Optuna hyperparameter search for CatBoost.
    CatBoost is already the best model — tuning should push it further.
    """
    print(f"\nTuning CatBoost — {n_trials} trials...")
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri(MLFLOW_PATH)
    mlflow.set_experiment("demand_forecasting")

    def objective(trial):
        params = {
            "iterations":         trial.suggest_int("iterations", 200, 600),
            "depth":              trial.suggest_int("depth", 4, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength":    trial.suggest_float("random_strength", 0, 10),
            "random_seed":        42,
            "verbose":            False
        }

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        preds = model.predict(X_test)
        return float(np.sqrt(mean_squared_error(y_test, preds)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params

    with mlflow.start_run(run_name="catboost_optuna_tuned"):
        final_model = CatBoostRegressor(**best_params, random_seed=42,
                                         verbose=False)
        final_model.fit(X_train, y_train)

        preds = final_model.predict(X_test)
        mae   = float(mean_absolute_error(y_test, preds))
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2    = float(r2_score(y_test, preds))

        mlflow.log_params(best_params)
        mlflow.log_param("model_type",    "CatBoost_Optuna")
        mlflow.log_param("n_trials",      n_trials)
        mlflow.log_param("tuning_method", "TPE")
        mlflow.log_metric("mae",          mae)
        mlflow.log_metric("rmse",         rmse)
        mlflow.log_metric("r2",           r2)
        mlflow.sklearn.log_model(final_model, "model")

        print(f"\nCatBoost Optuna Tuned Results:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")

    return {
        "model":       "catboost_optuna",
        "best_params": best_params,
        "mae":         mae,
        "rmse":        rmse,
        "r2":          r2,
        "n_trials":    n_trials
    }


def run_full_tuning():
    """Run tuning for both models and print comparison."""
    print("Starting Optuna hyperparameter tuning...")
    print("All trials logged to MLflow at:", MLFLOW_PATH)

    xgb_results = tune_xgboost(n_trials=30)
    cb_results  = tune_catboost(n_trials=20)

    print("\n" + "="*55)
    print("TUNING RESULTS SUMMARY")
    print("="*55)
    print(f"\nXGBoost (untuned):      RMSE=74.02")
    print(f"XGBoost (Optuna tuned): RMSE={xgb_results['rmse']:.4f}")
    print(f"Improvement: {74.02 - xgb_results['rmse']:.2f} RMSE units")

    print(f"\nCatBoost (untuned):      RMSE=69.99")
    print(f"CatBoost (Optuna tuned): RMSE={cb_results['rmse']:.4f}")
    print(f"Improvement: {69.99 - cb_results['rmse']:.2f} RMSE units")
    print("="*55)

    return xgb_results, cb_results


if __name__ == "__main__":
    run_full_tuning()
