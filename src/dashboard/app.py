import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT  = os.getenv("PROJECT_ROOT", "/Users/azeemkhalipha/mlops-retail-platform")
REPORTS_PATH  = f"{PROJECT_ROOT}/reports"
FEATURES_PATH = f"{PROJECT_ROOT}/data/features"

st.set_page_config(
    page_title="Demand Forecast Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-aware CSS — uses Streamlit's own CSS variables
# so it works in both light and dark mode automatically
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        font-size: 13px;
        padding: 6px 12px;
    }
    .status-bar {
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 14px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .status-warn {
        border-color: #d32f2f;
        color: #d32f2f;
        background: rgba(211, 47, 47, 0.08);
    }
    .status-ok {
        border-color: #388e3c;
        color: #388e3c;
        background: rgba(56, 142, 60, 0.08);
    }
</style>
""", unsafe_allow_html=True)


def load_drift_reports():
    files   = sorted(glob.glob(f"{REPORTS_PATH}/drift_summary_*.json"))
    reports = []
    for f in files:
        with open(f) as fp:
            reports.append(json.load(fp))
    return reports


def make_chart_layout(height=280, **kwargs):
    """
    Returns Plotly layout with transparent backgrounds
    so charts inherit the Streamlit theme automatically.
    Works in both light and dark mode.
    """
    return dict(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        **kwargs
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Demand Forecast Monitor")
    st.markdown(f"*{datetime.now().strftime('%d %b %Y, %H:%M')}*")
    st.markdown("---")
    st.markdown("**Actions**")

    run_drift   = st.button("Check drift now")
    run_retrain = st.button("Retrain model")

    if run_drift:
        with st.spinner("Running drift check..."):
            try:
                import sys
                sys.path.insert(0, PROJECT_ROOT)
                from src.monitoring.drift_detector import (
                    load_reference_data,
                    load_current_data,
                    generate_drift_report
                )
                os.makedirs(REPORTS_PATH, exist_ok=True)
                ref     = load_reference_data(FEATURES_PATH)
                curr    = load_current_data(FEATURES_PATH)
                summary = generate_drift_report(ref, curr, REPORTS_PATH)
                if summary["retrain_needed"]:
                    st.warning(
                        f"Drift detected — "
                        f"{summary['n_drifted_features']}/{summary['n_features']} "
                        f"features ({summary['drift_share']:.0%}). "
                        f"Retraining recommended."
                    )
                else:
                    st.success("No significant drift. Model looks healthy.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if run_retrain:
        with st.spinner("Retraining..."):
            try:
                import sys
                sys.path.insert(0, PROJECT_ROOT)
                from src.training.retrain import load_features, train_model
                df      = load_features()
                metrics = train_model(df)
                st.success(
                    f"Done. MAE {metrics['mae']:.1f} | "
                    f"RMSE {metrics['rmse']:.1f} | "
                    f"R² {metrics['r2']:.3f}"
                )
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Linear Regression")
    st.markdown("**Features**")
    st.markdown("7 lag/rolling features")
    st.markdown("**Retrain threshold**")
    st.markdown("50% features drifted")
    st.markdown("**Drift test**")
    st.markdown("Kolmogorov-Smirnov")


# ── Main content ──────────────────────────────────────────────────────────────
reports = load_drift_reports()

st.title("Demand Forecast Monitor")

if not reports:
    st.info("No drift reports yet. Run a drift check from the sidebar.")
    st.stop()

latest      = reports[-1]
report_date = datetime.strptime(
    latest["date"], "%Y%m%d"
).strftime("%d %b %Y")

st.markdown(f"Last report: **{report_date}**")

# Status bar — uses rgba backgrounds so readable in both themes
status_class = "status-warn" if latest["retrain_needed"] else "status-ok"
status_text  = "Retraining recommended" if latest["retrain_needed"] else "Model healthy"
st.markdown(
    f"<div class='status-bar {status_class}'>"
    f"<strong>{status_text}</strong> — "
    f"{latest['n_drifted_features']} of {latest['n_features']} features drifted "
    f"({latest['drift_share']:.0%})"
    f"</div>",
    unsafe_allow_html=True
)

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Drift share",      f"{latest['drift_share']:.0%}")
col2.metric("Features drifted", f"{latest['n_drifted_features']} / {latest['n_features']}")
col3.metric("Total reports",    len(reports))
col4.metric("Last checked",     report_date)

st.markdown("---")

# ── Charts row ────────────────────────────────────────────────────────────────
left, right = st.columns([1.2, 1])

with left:
    st.markdown("##### Drift share over time")

    df_reports = pd.DataFrame(reports)
    df_reports["date"] = pd.to_datetime(df_reports["date"], format="%Y%m%d")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_reports["date"],
        y=df_reports["drift_share"],
        mode="lines+markers",
        line=dict(color="#1976d2", width=2),
        marker=dict(size=6),
        name="Drift share"
    ))
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#d32f2f",
        line_width=1,
        annotation_text="Threshold",
        annotation_font_size=11
    )
    fig.update_layout(
        **make_chart_layout(),
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("##### KS statistic by feature")

    feature_data = [
        {
            "feature":      feat,
            "ks_statistic": s["ks_statistic"],
            "drifted":      s["drifted"]
        }
        for feat, s in latest["feature_stats"].items()
    ]
    df_feat = pd.DataFrame(feature_data).sort_values("ks_statistic")
    colors  = ["#d32f2f" if d else "#1976d2" for d in df_feat["drifted"]]

    fig2 = go.Figure(go.Bar(
        x=df_feat["ks_statistic"],
        y=df_feat["feature"],
        orientation="h",
        marker_color=colors
    ))
    fig2.add_vline(
        x=0.05,
        line_dash="dot",
        line_color="#999",
        line_width=1
    )
    fig2.update_layout(
        **make_chart_layout(),
        xaxis_title="KS statistic"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Feature table ─────────────────────────────────────────────────────────────
st.markdown("##### Feature drift details")

table_data = [
    {
        "Feature":      feat,
        "Ref mean":     s["ref_mean"],
        "Curr mean":    s["curr_mean"],
        "Change":       round(s["curr_mean"] - s["ref_mean"], 4),
        "KS statistic": s["ks_statistic"],
        "p-value":      s["p_value"],
        "Drifted":      "Yes" if s["drifted"] else "No"
    }
    for feat, s in latest["feature_stats"].items()
]
st.dataframe(
    pd.DataFrame(table_data),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ── Distribution comparison ───────────────────────────────────────────────────
st.markdown("##### Distribution: reference vs current")

selected = st.selectbox(
    "Feature",
    options=[d["Feature"] for d in table_data],
    label_visibility="collapsed"
)

df_ml     = pd.read_parquet(f"{FEATURES_PATH}/ml_features")
df_ml     = df_ml[[selected]].dropna()
split     = int(len(df_ml) * 0.7)
ref_vals  = df_ml.iloc[:split][selected]
curr_vals = df_ml.iloc[split:][selected]

fig3 = go.Figure()
fig3.add_trace(go.Histogram(
    x=ref_vals.sample(min(5000, len(ref_vals)), random_state=42),
    name="Reference",
    opacity=0.6,
    nbinsx=50,
    marker_color="#1976d2"
))
fig3.add_trace(go.Histogram(
    x=curr_vals.sample(min(5000, len(curr_vals)), random_state=42),
    name="Current",
    opacity=0.6,
    nbinsx=50,
    marker_color="#d32f2f"
))
fig3.update_layout(
    **make_chart_layout(height=300),
    barmode="overlay",
    xaxis_title=selected,
    yaxis_title="Count",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(fig3, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with PySpark · MLflow · FastAPI · "
    "Docker · Airflow · Streamlit · GitHub Actions"
)



# RAG chat component
import sys as _sys
_sys.path.insert(0, PROJECT_ROOT)
from chat_component import render_chat
render_chat(PROJECT_ROOT)
