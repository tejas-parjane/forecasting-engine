"""Reproducible backtest evaluation on the bundled sample dataset.

Runs the full forecasting pipeline with the documented showcase config
(daily data, 1,095 days 2021-2023, horizon 30, 4 expanding folds from 60%
train, sMAPE selection), refits the winning model on the full history for
in-sample residual diagnostics, and saves everything to
``results/backtest_sample.json``.

    python scripts/run_backtest.py

ARIMA order search makes this take a few minutes. No API key needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.sample_data import generate_business_demand  # noqa: E402
from app.evaluation.diagnostics import residual_diagnostics  # noqa: E402
from app.forecasting.pipeline import ForecastingPipeline, PipelineConfig  # noqa: E402
from app.models.registry import create_models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

CONFIG = PipelineConfig(horizon=30, backtest_folds=4, selection_metric="smape")


def winner_residuals(model_name: str, series: pd.Series) -> np.ndarray | None:
    """Refit the winning model on the full series and return in-sample residuals."""
    models = {m.name: m for m in create_models(series)}
    model = models.get(model_name)
    if model is None:
        return None
    model.fit(series)
    results = getattr(model, "results_", None)
    resid = getattr(results, "resid", None) if results is not None else None
    if resid is None:
        return None
    return np.asarray(resid, dtype=float)


def main() -> None:
    df = generate_business_demand(n_days=1095, seed=42)
    print(f"dataset: {len(df)} daily rows, {df['date'].min()} to {df['date'].max()}")

    result = ForecastingPipeline(config=CONFIG).run(df)
    backtest = result.backtest
    series = pd.Series(df["target"].to_numpy(), index=pd.to_datetime(df["date"]))

    diagnostics = None
    resid = winner_residuals(backtest["best_model"], series)
    if resid is not None:
        diagnostics = residual_diagnostics(resid)

    payload = {
        "dataset": {"rows": len(df), "granularity": "daily",
                    "start": str(df["date"].min()), "end": str(df["date"].max()), "seed": 42},
        "config": {"horizon": CONFIG.horizon, "backtest_folds": CONFIG.backtest_folds,
                   "selection_metric": CONFIG.selection_metric,
                   "initial_train_frac": CONFIG.initial_train_frac},
        "backtest": backtest,
        "winner_diagnostics": {"model": backtest["best_model"], "diagnostics": diagnostics},
    }
    out = ROOT / "results" / "backtest_sample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out}")

    print("\n=== BACKTEST (expanding window, sMAPE selection) ===")
    for m in sorted(backtest["models"], key=lambda x: (x["rank"] or 99)):
        agg = m["aggregated"]
        print(f"  rank {m['rank']}: {m['model_name']:15s} sMAPE={agg.get('smape')} "
              f"MAE={agg.get('mae')} RMSE={agg.get('rmse')} WAPE={agg.get('wape')}")
    print(f"winner: {backtest['best_model']} ({backtest['best_score']})")
    if diagnostics:
        print("winner residuals:", json.dumps(diagnostics, indent=1)[:600])


if __name__ == "__main__":
    main()
