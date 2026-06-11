"""
SHAP-based model explainability.

SHAP (SHapley Additive exPlanations) explains individual predictions
by calculating each feature's contribution.

Example: Model predicted 45 units because:
  - qty_lag_7 contributed +12 units (sales were high last week)
  - qty_rolling_avg_30 contributed +8 units (strong monthly trend)
  - daily_revenue contributed -3 units (revenue was below average)
  - ... and so on for all 7 features
"""

import os
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
MODEL_PATH   = f"{PROJECT_ROOT}/models/demand_forecast_v1.pkl"
PLOTS_PATH   = f"{PROJECT_ROOT}/reports/shap"

FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_30",
    "qty_rolling_avg_7", "qty_rolling_avg_30",
    "qty_rolling_std_7", "daily_revenue"
]


def load_model():
    """Load the trained model from pickle."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_explainer(model, X_background: pd.DataFrame):
    """
    Create a SHAP explainer for the model.

    We use KernelExplainer which works with any model type —
    sklearn, XGBoost, LightGBM, CatBoost — without needing
    model-specific explainers.

    X_background is a small sample of training data that SHAP
    uses as a baseline to measure feature contributions against.
    """
    # Use 100 background samples for speed
    background = shap.kmeans(X_background, 50)
    return shap.KernelExplainer(model.predict, background)


def explain_prediction(
    features: dict,
    X_background: pd.DataFrame
) -> dict:
    """
    Explain a single prediction.

    Returns:
    - prediction: the model's predicted value
    - base_value: what the model predicts with no information (mean)
    - shap_values: how much each feature pushed the prediction up or down
    - feature_contributions: human-readable dict of contributions
    """
    model = load_model()
    df    = pd.DataFrame([features])

    # Get prediction
    prediction = float(model.predict(df)[0])

    # Get SHAP values
    explainer   = get_explainer(model, X_background)
    shap_values = explainer.shap_values(df)

    # Build human-readable contributions
    contributions = {}
    for i, col in enumerate(FEATURE_COLS):
        contributions[col] = {
            "feature_value": round(float(features[col]), 4),
            "shap_value":    round(float(shap_values[0][i]), 4),
            "direction":     "increases" if shap_values[0][i] > 0 else "decreases",
            "impact":        "high" if abs(shap_values[0][i]) > 2 else
                            "medium" if abs(shap_values[0][i]) > 0.5 else "low"
        }

    # Sort by absolute SHAP value (most impactful first)
    contributions = dict(
        sorted(
            contributions.items(),
            key=lambda x: abs(x[1]["shap_value"]),
            reverse=True
        )
    )

    return {
        "prediction":          round(prediction, 2),
        "base_value":          round(float(explainer.expected_value), 2),
        "feature_contributions": contributions,
        "top_feature":         list(contributions.keys())[0],
        "explanation":         _build_explanation(prediction, contributions)
    }


def _build_explanation(prediction: float, contributions: dict) -> str:
    """Build a plain-English explanation of the prediction."""
    top_3 = list(contributions.items())[:3]
    parts = []
    for feat, info in top_3:
        direction = "increased" if info["shap_value"] > 0 else "decreased"
        parts.append(
            f"{feat} {direction} the forecast by "
            f"{abs(info['shap_value']):.1f} units"
        )

    return (
        f"Predicted demand: {prediction:.1f} units. "
        f"Key drivers: {'; '.join(parts)}."
    )


def generate_summary_plot(X_sample: pd.DataFrame, save_path: str = None):
    """
    Generate a SHAP summary plot showing feature importance
    across many predictions.

    This is the most useful plot for understanding overall
    model behaviour — not just one prediction.
    """
    model     = load_model()
    explainer = get_explainer(model, X_sample)

    # Compute SHAP values for a sample of rows
    sample      = X_sample.sample(min(200, len(X_sample)), random_state=42)
    shap_values = explainer.shap_values(sample)

    os.makedirs(PLOTS_PATH, exist_ok=True)
    save_path = save_path or f"{PLOTS_PATH}/shap_summary.png"

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        sample,
        feature_names=FEATURE_COLS,
        show=False
    )
    plt.title("SHAP Feature Importance — Impact on Demand Forecast")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved: {save_path}")
    return save_path


def generate_waterfall_plot(features: dict, X_background: pd.DataFrame,
                            save_path: str = None):
    """
    Generate a waterfall plot for a single prediction.
    Shows how each feature pushes the prediction up or down
    from the base value.
    """
    model       = load_model()
    df          = pd.DataFrame([features])
    explainer   = get_explainer(model, X_background)
    shap_values = explainer.shap_values(df)

    os.makedirs(PLOTS_PATH, exist_ok=True)
    save_path = save_path or f"{PLOTS_PATH}/shap_waterfall.png"

    # Manual waterfall chart since SHAP's built-in requires newer API
    base_val  = float(explainer.expected_value)
    sv        = shap_values[0]
    fv        = [features[c] for c in FEATURE_COLS]

    fig, ax = plt.subplots(figsize=(10, 6))

    cumulative = base_val
    colors     = []
    positions  = []
    widths     = []
    labels     = []

    for i, (col, val) in enumerate(zip(FEATURE_COLS, sv)):
        colors.append("#d32f2f" if val > 0 else "#1976d2")
        positions.append(cumulative + (val / 2 if val > 0 else val / 2))
        widths.append(abs(val))
        labels.append(f"{col}\n({fv[i]:.1f})")
        cumulative += val

    ax.barh(
        range(len(FEATURE_COLS)),
        [s for s in sv],
        left=[base_val + sum(sv[:i]) for i in range(len(sv))],
        color=colors,
        height=0.6
    )
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels(FEATURE_COLS, fontsize=11)
    ax.axvline(x=base_val, color="black", linestyle="--", alpha=0.5,
               label=f"Base value: {base_val:.1f}")
    ax.axvline(x=cumulative, color="green", linestyle="-", alpha=0.7,
               label=f"Prediction: {cumulative:.1f}")
    ax.set_xlabel("Demand forecast (units)")
    ax.set_title("SHAP Waterfall — Feature Contributions to This Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Waterfall plot saved: {save_path}")
    return save_path


if __name__ == "__main__":
    # Load features for background
    features_path = f"{PROJECT_ROOT}/data/features"
    df_bg = pd.read_parquet(f"{features_path}/ml_features")
    df_bg = df_bg[FEATURE_COLS].dropna().sample(500, random_state=42)

    # Test single prediction explanation
    sample_features = {
        "qty_lag_1":          10.0,
        "qty_lag_7":          8.0,
        "qty_lag_30":         9.0,
        "qty_rolling_avg_7":  9.5,
        "qty_rolling_avg_30": 9.0,
        "qty_rolling_std_7":  1.2,
        "daily_revenue":      45.0
    }

    print("Explaining single prediction...")
    result = explain_prediction(sample_features, df_bg)

    print(f"\nPrediction:  {result['prediction']} units")
    print(f"Base value:  {result['base_value']} units")
    print(f"Top feature: {result['top_feature']}")
    print(f"\nExplanation: {result['explanation']}")
    print(f"\nFeature contributions:")
    for feat, info in result["feature_contributions"].items():
        arrow = "▲" if info["shap_value"] > 0 else "▼"
        print(f"  {arrow} {feat:25s}: {info['shap_value']:+.4f} ({info['impact']} impact)")

    print("\nGenerating summary plot...")
    generate_summary_plot(df_bg)

    print("\nGenerating waterfall plot...")
    generate_waterfall_plot(sample_features, df_bg)

    print("\nDone. Check reports/shap/ for plots.")
