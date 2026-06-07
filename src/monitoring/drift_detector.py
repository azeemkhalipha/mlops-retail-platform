import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from scipy import stats

# The 7 features we monitor for drift
# Same features the model was trained on
FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_30",
    "qty_rolling_avg_7", "qty_rolling_avg_30",
    "qty_rolling_std_7", "daily_revenue"
]

# If more than 50% of features drift, trigger retraining
DRIFT_THRESHOLD = 0.5


def load_reference_data(features_path: str) -> pd.DataFrame:
    """
    Load the first 70% of data as reference (training distribution).
    This is what the model learned from — the baseline.
    """
    df    = pd.read_parquet(f"{features_path}/ml_features")
    df    = df[FEATURE_COLS].dropna()
    split = int(len(df) * 0.7)
    return df.iloc[:split]


def load_current_data(features_path: str) -> pd.DataFrame:
    """
    Load the last 30% of data as current (incoming distribution).
    This simulates new data arriving in production.
    """
    df    = pd.read_parquet(f"{features_path}/ml_features")
    df    = df[FEATURE_COLS].dropna()
    split = int(len(df) * 0.7)
    return df.iloc[split:]


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame
) -> dict:
    """
    Run Kolmogorov-Smirnov test on each feature.

    KS test compares two distributions without assuming
    they follow any particular shape (non-parametric).

    Returns:
    - ks_statistic: how different the distributions are (0=same, 1=completely different)
    - p_value: probability of seeing this difference by chance
    - drifted: True if p_value < 0.05 (statistically significant difference)
    """
    drifted_features = []
    feature_stats    = {}

    for col in FEATURE_COLS:
        ref_values  = reference[col].dropna().values
        curr_values = current[col].dropna().values

        # KS test: are these two samples from the same distribution?
        ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)

        # p_value < 0.05 = 95% confident the distributions differ
        is_drifted = bool(p_value < 0.05)

        feature_stats[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value":      round(float(p_value), 4),
            "drifted":      is_drifted,
            "ref_mean":     round(float(ref_values.mean()), 4),
            "curr_mean":    round(float(curr_values.mean()), 4),
            "ref_std":      round(float(ref_values.std()), 4),
            "curr_std":     round(float(curr_values.std()), 4)
        }

        if is_drifted:
            drifted_features.append(col)

    n_drifted     = len(drifted_features)
    n_features    = len(FEATURE_COLS)
    drift_share   = float(n_drifted / n_features)
    dataset_drift = bool(drift_share > DRIFT_THRESHOLD)

    return {
        "feature_stats":      feature_stats,
        "drifted_features":   drifted_features,
        "n_drifted_features": int(n_drifted),
        "n_features":         int(n_features),
        "drift_share":        drift_share,
        "dataset_drift":      dataset_drift
    }


def _save_html_report(summary: dict, drift_results: dict, path: str):
    """Generate a simple HTML drift report."""
    rows = ""
    for col, s in drift_results["feature_stats"].items():
        color  = "#ff4444" if s["drifted"] else "#44bb44"
        status = "DRIFTED" if s["drifted"] else "OK"
        rows += f"""
        <tr>
            <td>{col}</td>
            <td>{s['ref_mean']}</td>
            <td>{s['curr_mean']}</td>
            <td>{s['ks_statistic']}</td>
            <td>{s['p_value']}</td>
            <td style='color:{color};font-weight:bold'>{status}</td>
        </tr>"""

    alert_class = "red"   if summary["dataset_drift"] else "green"
    alert_text  = "DRIFT DETECTED" if summary["dataset_drift"] else "NO DRIFT"

    html = f"""<html>
    <head><title>Drift Report {summary['date']}</title>
    <style>
        body  {{ font-family: Arial; padding: 20px; background: #1a1a2e; color: #eee; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #444; padding: 10px; text-align: left; }}
        th    {{ background: #16213e; color: #0f3460; color: white; }}
        tr:nth-child(even) {{ background: #16213e; }}
        .alert {{ padding: 15px; margin: 15px 0; border-radius: 8px; font-size: 18px; }}
        .red   {{ background: #3d0000; border-left: 6px solid #ff4444; }}
        .green {{ background: #003d00; border-left: 6px solid #44bb44; }}
        h1 {{ color: #e94560; }}
        h2 {{ color: #0f3460; color: #a8dadc; }}
    </style></head>
    <body>
    <h1>Data Drift Report</h1>
    <p>Date: {summary['date']} | Features monitored: {summary['n_features']}</p>
    <div class='alert {alert_class}'>
        <strong>{alert_text}</strong> —
        {summary['n_drifted_features']}/{summary['n_features']} features drifted
        ({summary['drift_share']:.1%}) |
        Retrain needed: {'YES' if summary['retrain_needed'] else 'NO'}
    </div>
    <h2>Feature Statistics</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Ref Mean</th>
            <th>Curr Mean</th>
            <th>KS Statistic</th>
            <th>P-Value</th>
            <th>Status</th>
        </tr>
        {rows}
    </table>
    </body></html>"""

    with open(path, "w") as f:
        f.write(html)


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reports_path: str
) -> dict:
    """
    Run drift detection, save HTML report and JSON summary.
    Returns summary dict with all drift metrics.
    """
    today         = datetime.today().strftime("%Y%m%d")
    drift_results = detect_drift(reference, current)

    summary = {
        "date":               today,
        "dataset_drift":      drift_results["dataset_drift"],
        "drift_share":        drift_results["drift_share"],
        "n_drifted_features": drift_results["n_drifted_features"],
        "n_features":         drift_results["n_features"],
        "drifted_features":   drift_results["drifted_features"],
        "retrain_needed":     bool(drift_results["drift_share"] > DRIFT_THRESHOLD),
        "feature_stats":      drift_results["feature_stats"]
    }

    # Save JSON summary — used by Airflow DAG to decide retraining
    summary_path = f"{reports_path}/drift_summary_{today}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save HTML report — human-readable visual report
    report_path = f"{reports_path}/drift_report_{today}.html"
    _save_html_report(summary, drift_results, report_path)

    print(f"\nDrift Summary:")
    print(f"  Date:                   {today}")
    print(f"  Dataset drift detected: {summary['dataset_drift']}")
    print(f"  Drift share:            {summary['drift_share']:.2%}")
    print(f"  Drifted features:       {summary['n_drifted_features']}/{summary['n_features']}")
    print(f"  Drifted columns:        {summary['drifted_features']}")
    print(f"  Retrain needed:         {summary['retrain_needed']}")
    print(f"  HTML report:            {report_path}")
    print(f"  JSON summary:           {summary_path}")

    return summary


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

    PROJECT_ROOT  = os.getenv("PROJECT_ROOT")
    FEATURES_PATH = f"{PROJECT_ROOT}/data/features"
    REPORTS_PATH  = f"{PROJECT_ROOT}/reports"

    os.makedirs(REPORTS_PATH, exist_ok=True)

    print("Loading reference data (first 70%)...")
    reference = load_reference_data(FEATURES_PATH)
    print(f"Reference shape: {reference.shape}")

    print("\nLoading current data (last 30%)...")
    current = load_current_data(FEATURES_PATH)
    print(f"Current shape: {current.shape}")

    summary = generate_drift_report(reference, current, REPORTS_PATH)
    print("\nDone.")
