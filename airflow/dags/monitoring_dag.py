from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

PROJECT_ROOT = "/Users/azeemkhalipha/mlops-retail-platform"

default_args = {
    "owner":       "mlops",
    "retries":     1,
    "retry_delay": timedelta(minutes=5),
    "start_date":  datetime(2024, 1, 1)
}


def check_drift(**context):
    """
    Task 1: Run drift detection.
    Saves HTML report and JSON summary to reports/ folder.
    Pushes drift summary to XCom so next task can read it.
    XCom = Airflow's way of passing data between tasks.
    """
    sys.path.insert(0, PROJECT_ROOT)
    from src.monitoring.drift_detector import (
        load_reference_data,
        load_current_data,
        generate_drift_report
    )

    features_path = f"{PROJECT_ROOT}/data/features"
    reports_path  = f"{PROJECT_ROOT}/reports"
    os.makedirs(reports_path, exist_ok=True)

    reference = load_reference_data(features_path)
    current   = load_current_data(features_path)
    summary   = generate_drift_report(reference, current, reports_path)

    # Push result to XCom so decide_retrain task can read it
    context["ti"].xcom_push(key="drift_summary", value=summary)
    print(f"Drift check complete — retrain_needed: {summary['retrain_needed']}")
    return summary


def decide_retrain(**context):
    """
    Task 2: Read drift result and decide whether to retrain.
    This is the decision gate — only retrains when necessary.
    """
    summary = context["ti"].xcom_pull(
        task_ids="check_drift",
        key="drift_summary"
    )
    retrain_needed = summary.get("retrain_needed", False)
    drift_share    = summary.get("drift_share", 0)

    print(f"Drift share:    {drift_share:.2%}")
    print(f"Retrain needed: {retrain_needed}")
    return retrain_needed


def run_retrain(**context):
    """
    Task 3a: Retrain the model.
    Only runs if decide_retrain returned True.
    """
    sys.path.insert(0, PROJECT_ROOT)
    from src.training.retrain import load_features, train_model

    df      = load_features()
    metrics = train_model(df)
    print(f"Retraining complete: {metrics}")
    return metrics


def skip_retrain(**context):
    """
    Task 3b: Log that no retraining was needed.
    Runs when drift is below threshold.
    """
    print("No significant drift detected — model is healthy, skipping retraining.")


with DAG(
    dag_id="monitoring_and_retraining",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    description="Daily drift monitoring with automatic retraining trigger"
) as dag:

    check_drift_task = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
        provide_context=True
    )

    decide_task = PythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain,
        provide_context=True
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=run_retrain,
        provide_context=True
    )

    skip_task = PythonOperator(
        task_id="skip_retrain",
        python_callable=skip_retrain,
        provide_context=True
    )

    # DAG flow: check drift → decide → retrain OR skip
    check_drift_task >> decide_task >> [retrain_task, skip_task]
