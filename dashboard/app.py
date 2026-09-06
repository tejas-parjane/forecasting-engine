"""AI-Powered Forecasting & Decision Intelligence Engine — dashboard.

Single-page Streamlit application integrating the forecasting pipeline and
the AI decision-intelligence layer.

Screen answers, in order:
  1. What is happening?      (historical analysis)
  2. What is expected?       (forecast)
  3. How confident are we?   (uncertainty)
  4. Why is it happening?    (explainability)
  5. What should we do?      (decision intelligence)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# Always prepend the repo root: a conditional insert is skipped when ROOT is
# already on sys.path (e.g. via PYTHONPATH) yet positioned after the script
# dir, which would resolve `import app` to this script (module `app`) instead
# of the real app/ package, raising "'app' is not a package".
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.data.sample_data import generate_business_demand  # noqa: E402
from app.forecasting.pipeline import ForecastingPipeline, PipelineConfig  # noqa: E402

st.set_page_config(
    page_title="Forecasting & Decision Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #0e1117;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.8; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; color: #fafafa; }
    a { color: #4aa3ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

ACCENT = "#4aa3ff"
POS = "#2bb968"
NEG = "#ef5350"
NEU = "#9aa4b2"


# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    path = settings.resolve_sample_data()
    if path.exists():
        return pd.read_csv(path)
    return generate_business_demand(seed=settings.sample_data_seed)


def fmt(v: float | None, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.{nd}f}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Forecasting Engine")
    st.caption("AI-Powered Forecasting & Decision Intelligence")

    source = st.radio("Data source", ["Sample dataset", "Upload CSV"], index=0)
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload time-series CSV", type=["csv", "xlsx"])
        if uploaded is None:
            st.info("Upload a CSV with a date column and a target column.")
            st.stop()

    st.divider()
    st.subheader("Configuration")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        date_candidates = [
            c for c in raw.columns if "date" in c.lower() or "time" in c.lower() or "day" in c.lower()
        ]
        target_candidates = [
            c for c in raw.columns if "target" in c.lower() or "value" in c.lower()
            or "demand" in c.lower() or "sales" in c.lower() or raw[c].dtype.kind == "f"
        ]
        numeric_cols = list(raw.select_dtypes(include="number").columns)
        if not date_candidates and not numeric_cols:
            st.error("No identifiable columns. Expected a date column and a numeric target.")
            st.stop()
        date_col = st.selectbox("Date column", options=list(raw.columns), index=0)
        target_col = st.selectbox(
            "Target column",
            options=(date_candidates + numeric_cols if date_candidates else numeric_cols) or list(raw.columns),
        )
    else:
        raw = None
        date_col = "date"
        target_col = "target"

    horizon = st.select_slider(
        "Forecast horizon (periods)", options=[7, 14, 30, 60, 90], value=30,
    )
    selection_metric = st.selectbox(
        "Model selection metric", ["smape", "mae", "rmse", "wape", "mape"], index=0,
    )
    use_ai = st.toggle(
        "Use AI layer",
        value=settings.ai_enabled,
        disabled=not settings.ai_enabled,
        help="Requires AI_ENABLED=true and OPENAI_API_KEY in .env. When off, deterministic rule-based briefs are used.",
    )

    run = st.button("Run pipeline", type="primary", use_container_width=True)

    st.caption(f"AI layer: {'enabled' if settings.ai_enabled else 'off — deterministic rule-based briefs are used'}")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Forecasting & Decision Intelligence")

if not run and "pipeline_result" not in st.session_state:
    st.markdown(
        """
        This application turns historical business time-series data into:
        **validated forecasts → uncertainty estimates → driver explanations →
        AI-assisted decision briefs**.

        Configure the data source and horizon in the sidebar, then press
        **Run pipeline**. The system runs expanding-window backtests over
        five model families, selects the best by validated error, and produces
        a probabilistic forecast with an interpretation layer.
        """
    )
    if uploaded is None and source == "Sample dataset":
        preview = load_default_data()
        st.markdown("**Sample data preview**")
        st.dataframe(preview.head(), use_container_width=True)
    st.stop()

# --------------------------------------------------------------------------
# Pipeline execution
# --------------------------------------------------------------------------
with st.spinner("Running validation, EDA, backtesting, forecasting, and explanation…"):
    if uploaded is not None:
        df = raw
    else:
        df = load_default_data()

    cfg = PipelineConfig(
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        selection_metric=selection_metric,
        backtest_folds=4,
        initial_train_frac=0.6,
        include_xgboost=True,
    )
    result = ForecastingPipeline(config=cfg).run(df, horizon=horizon)
    st.session_state.pipeline_result = result

result = st.session_state.pipeline_result
fc = result.forecast
eda = result.eda
bt = result.backtest
exp = result.explainability
report = result.report

# --------------------------------------------------------------------------
# KPI Overview
# --------------------------------------------------------------------------
expected = fc.get("expected_change", {})
last_actual = result.eda["summary_stats"]["recent_value"]
fc_mean = float(np.mean(fc["forecast"]))
fc_end = float(fc["forecast"][-1])
pct = expected.get("pct_change")

direction_color = {"increase": POS, "decrease": NEG, "flat": NEU}.get(expected.get("direction"), NEU)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Latest actual", fmt(last_actual))
c2.metric(
    "Forecasted avg",
    fmt(fc_mean),
    delta=f"{pct:+.1f}% vs latest" if pct is not None else None,
    delta_color="normal",
)
c3.metric("Horizon", f"{horizon} periods", help="Forecast length.")
c4.metric("Selected model", fc["model"].replace("_", " ").title())
c5.metric("Data quality", f"{report['quality_score']:.0f}/100")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tabs = st.tabs(
    ["📊 Historical", "⚖️ Models", "🎯 Forecast", "🔍 Explain", "🧠 Decision"]
)

# --- Historical ---
with tabs[0]:
    h1, h2 = st.columns([2, 2])
    hist_dates = fc["historical"]["dates"]
    hist_vals = fc["historical"]["values"]

    with h1:
        st.subheader("Historical series & rolling statistics")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_vals, name="Actual", line=dict(color=ACCENT, width=1.4)))
        s = pd.Series(hist_vals, index=pd.to_datetime(hist_dates))
        fig.add_trace(
            go.Scatter(
                x=hist_dates[27:], y=s.rolling(28).mean().dropna(), name="28-day mean",
                line=dict(color=POS, width=1.6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist_dates[27:], y=s.rolling(28).std().dropna(), name="28-day std",
                line=dict(color="#f9a825", width=1.2, dash="dot"),
            )
        )
        # anomalies
        dates_s = pd.to_datetime(hist_dates)
        vals_s = np.array(hist_vals)
        q1, q3 = np.percentile(vals_s, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            anom_mask = (vals_s < q1 - 3 * iqr) | (vals_s > q3 + 3 * iqr)
            fig.add_trace(
                go.Scatter(
                    x=dates_s[anom_mask], y=vals_s[anom_mask], name="Anomaly",
                    mode="markers", marker=dict(color=NEG, size=7, symbol="x"),
                )
            )
        fig.update_layout(height=420, margin=dict(t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with h2:
        st.subheader("Seasonal decomposition")
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            s = pd.Series(hist_vals, index=pd.to_datetime(hist_dates))
            decomp = seasonal_decompose(s, model="additive", period=7, extrapolate_trend="freq")
            fig2 = make_subplots_fig(hist_dates, decomp)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info("Decomposition unavailable for this dataset.")

        st.subheader("Distribution of target")
        fig3 = go.Figure(go.Histogram(x=hist_vals, nbinsx=40, marker_color=ACCENT))
        fig3.update_layout(height=200, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Trend**")
        t = eda["trend"]
        color = {"increasing": POS, "decreasing": NEG, "stable": NEU}.get(t["direction"], NEU)
        st.markdown(f"<span style='color:{color};font-weight:600'>{t['direction'].upper()}</span>", unsafe_allow_html=True)
        if t.get("slope_pct_per_period") is not None:
            st.caption(f"slope ≈ {t['slope_pct_per_period']:.3f}% per period")
        st.caption(f"p-value: {t.get('p_value')}")
    with colB:
        st.markdown("**Seasonality**")
        seas = eda["seasonality"]
        if seas.get("detected"):
            st.markdown(f"**{seas['seasonality_type']}** pattern")
            st.caption(f"period {seas.get('period')}, strength {seas.get('strength')}")
        else:
            st.markdown("No strong seasonal pattern")
    with colC:
        st.markdown("**Volatility & anomalies**")
        v = eda["volatility"]
        st.caption(f"recent std: {fmt(v.get('recent_std'), 1)} · trend: {v.get('trend')}")
        st.caption(f"anomalies: {eda['anomalies']['count']} detected ({eda['anomalies']['method']})")

# --- Models ---
with tabs[1]:
    st.subheader("Model comparison — expanding-window backtest")
    st.caption(
        "Each model is re-fitted on a growing window of the past and evaluated on the "
        "following held-out period. No random train/test split is used. Scores are "
        "aggregated across folds weighted by fold size."
    )
    rows = []
    for m in bt["models"]:
        agg = m["aggregated"]
        rows.append(
            {
                "model": m["model_name"],
                "rank": m.get("rank"),
                "MAE": agg.get("mae"),
                "RMSE": agg.get("rmse"),
                "MAPE%": agg.get("mape"),
                "sMAPE%": agg.get("smape"),
                "WAPE%": agg.get("wape"),
                "valid folds": f"{agg.get('n_folds_valid')}/{agg.get('n_folds_total')}",
            }
        )
    comp = pd.DataFrame(rows).sort_values("rank", na_position="last")
    st.dataframe(
        comp,
        use_container_width=True,
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "sMAPE%": st.column_config.NumberColumn(format="%.2f"),
            "WAPE%": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    with st.expander("Fold detail"):
        for m in bt["models"]:
            built = False
            for f in m["scores_per_fold"]:
                if not built:
                    st.markdown(f"**{m['model_name']}**")
                    built = True
                st.caption(
                    f"fold {f['fold']}: train {f['train_start']} {f['train_end']} → "
                    f"val {f['val_start']} {f['val_end']} — "
                    f"sMAPE {f['metrics'].get('smape')}, MAE {f['metrics'].get('mae')}"
                )

    st.divider()
    st.markdown("**Metric caveats**")
    st.markdown(
        "- **MAE** — average absolute error in target units; robust to outliers.\n"
        "- **RMSE** — penalizes large errors more than MAE; sensitive to outliers.\n"
        "- **MAPE** — percentage error; undefined when actuals are zero, asymmetric.\n"
        "- **sMAPE** — symmetric percentage error; bounded in (−200, 200) but can be "
        "misleading near zero.\n"
        "- **WAPE** — sum of absolute errors / sum of actuals; stable for low-volume series."
    )

# --- Forecast ---
with tabs[2]:
    st.subheader(f"Forecast — {fc['model'].replace('_', ' ').title()} ({fc['method']})")
    lower = fc.get("lower")
    upper = fc.get("upper")

    x = hist_dates
    y = hist_vals
    fx = fc["forecast_dates"]
    fy = fc["forecast"]
    fl = lower if lower else [None] * len(fy)
    fu = upper if upper else [None] * len(fy)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=x, y=y, name="Observed", line=dict(color="#8b95a5", width=1.3)))
    fig4.add_trace(go.Scatter(x=fx, y=fy, name="Forecast", line=dict(color=ACCENT, width=2.4)))
    if fl and fu:
        fig4.add_trace(
            go.Scatter(
                x=fx + fx[::-1], y=fu + fl[::-1], fill="toself",
                fillcolor="rgba(74,163,255,0.15)", line=dict(width=0), name="95% interval",
            )
        )
    # level lines
    fig4.add_hline(y=last_actual, line_dash="dot", line_color="#f9a825", line_width=1,
                   annotation_text=f"last actual {last_actual:,.0f}")
    fig4.update_layout(height=440, margin=dict(t=20, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig4, use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Expected at horizon end", fmt(fy[-1]), delta=f"{pct:+.1f}%" if pct is not None else None)
    b2.metric("Horizon mean", fmt(fc_mean))
    if fl and fu:
        b3.metric("95% lower (end)", fmt(fl[-1]))
        b4.metric("95% upper (end)", fmt(fu[-1]))
    else:
        b3.metric("95% lower (end)", "—")
        b4.metric("95% upper (end)", "—")

    st.caption(
        "Uncertainty: prediction intervals widen with horizon — distant-period "
        f"forecasts are more uncertain than near-term ones. Method: {fc['metadata'].get('uncertainty_method', fc['method'])}"
    )

    with st.expander("Forecast table"):
        table = pd.DataFrame(
            {
                "date": fx,
                "forecast": fy,
                "lower": fl if fl else np.nan,
                "upper": fu if fu else np.nan,
            }
        )
        st.dataframe(table, use_container_width=True)

# --- Explain ---
with tabs[3]:
    e1, e2 = st.columns([3, 2])
    with e1:
        st.subheader("Forecast drivers")
        st.caption("Signals are statistically *associated with* the forecast. No causality is implied.")
        for d in exp["drivers"]:
            dcolor = {"positive": POS, "negative": NEG}.get(d.get("direction"), NEU)
            st.markdown(
                f"<span style='color:{dcolor};font-weight:600'>{d.get('direction','neutral').upper()}</span> "
                f"· **{d['driver']}** — {d['signal']}",
                unsafe_allow_html=True,
            )
    with e2:
        st.subheader("Model & uncertainty")
        st.markdown(f"**Selected model:** {exp['model']}")
        st.markdown(f"**Method:** {fc['method']}")
        st.markdown(f"**Horizon:** {fc['horizon']} periods")
        st.markdown(f"**Interval band:** {len(fu) if fu else 0} points at 95% coverage")
        st.caption(exp.get("uncertainty_note", ""))

    st.divider()
    st.subheader("Feature importance")
    if exp["feature_importance"]:
        imp = exp["feature_importance"]
        imp_df = pd.DataFrame(imp)[["feature", "importance"]].head(12)
        fig5 = go.Figure(
            go.Bar(x=imp_df["importance"], y=imp_df["feature"], orientation="h", marker_color=ACCENT)
        )
        fig5.update_layout(height=380, margin=dict(t=10, b=10), xaxis_title="importance")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Feature importance is available only for tree-based (XGBoost) models.")

# --- Decision ---
with tabs[4]:
    from app.ai.decision_engine import DecisionEngine

    engine = DecisionEngine(
        api_key=settings.openai_api_key,
        model=settings.ai_model,
        temperature=settings.ai_temperature,
        max_tokens=settings.ai_max_tokens,
    )
    structured = engine.build_structured_input(
        forecast_payload=fc,
        eda=eda,
        explainability=exp,
        report=report,
        backtest=bt,
    )
    use_llm = use_ai and settings.ai_enabled
    with st.spinner("Generating decision brief…" if use_llm else ""):
        brief = engine.generate(structured, use_llm=use_llm)

    st.subheader("Decision brief")
    if not use_llm:
        st.caption("Rule-based brief (no LLM configured). Enable the AI layer via OPENAI_API_KEY for LLM interpretation.")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Summary")
        st.markdown(brief.forecast_summary)
        st.markdown("#### Risk assessment")
        st.markdown(brief.risk_assessment)
    with d2:
        st.markdown("#### Driver analysis")
        st.markdown(brief.driver_analysis)
        st.markdown("#### Recommended actions")
        st.markdown(
            f"<div style='border-left:4px solid {POS};padding-left:12px'>{brief.business_recommendation}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Structured input to AI layer"):
        import json

        st.json(structured)


def make_subplots_fig(dates, decomp):
    import plotly.subplots as sp

    fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=dates, y=decomp.observed, name="observed", line=dict(color=ACCENT, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=dates, y=decomp.trend, name="trend", line=dict(color=POS, width=1.4)), 2, 1)
    fig.add_trace(go.Scatter(x=dates, y=decomp.seasonal, name="seasonal", line=dict(color="#f9a825", width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=dates, y=decomp.resid, name="residual", line=dict(color="#9aa4b2", width=1)), 4, 1)
    fig.update_layout(height=560, margin=dict(t=10, b=10), legend=dict(orientation="h"))
    return fig