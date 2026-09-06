"""High-level forecasting pipeline.

Orchestrates: validate → preprocess → explore → backtest → select →
retrain on full history → generate forecast with uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.eda import EDAAnalyzer
from app.data.validator import DataValidator
from app.evaluation.backtesting import Backtester
from app.models.registry import create_models


@dataclass
class PipelineConfig:
    date_col: str = "date"
    target_col: str = "target"
    horizon: int = 30
    backtest_folds: int = 4
    selection_metric: str = "smape"
    initial_train_frac: float = 0.6
    include_xgboost: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_col": self.date_col,
            "target_col": self.target_col,
            "horizon": self.horizon,
            "backtest_folds": self.backtest_folds,
            "selection_metric": self.selection_metric,
            "initial_train_frac": self.initial_train_frac,
            "include_xgboost": self.include_xgboost,
        }


@dataclass
class PipelineResult:
    report: dict[str, Any]
    eda: dict[str, Any]
    backtest: dict[str, Any]
    forecast: dict[str, Any]
    explainability: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "eda": self.eda,
            "backtest": self.backtest,
            "forecast": self.forecast,
            "explainability": self.explainability,
            "config": self.config,
        }


class ForecastingPipeline:
    """End-to-end forecasting workflow."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(
        self,
        df: pd.DataFrame,
        frequency: str | None = None,
        horizon: int | None = None,
    ) -> PipelineResult:
        cfg = self.config
        if horizon is not None:
            cfg.horizon = horizon

        # 1. Validate
        validator = DataValidator(date_col=cfg.date_col, target_col=cfg.target_col)
        report = validator.validate(df).to_dict()

        # 2. Preprocess
        from app.data.preprocessing import Preprocessor

        pre = Preprocessor(date_col=cfg.date_col, target_col=cfg.target_col)
        prepped = pre.preprocess(df, frequency=frequency)
        freq = frequency or prepped.frequency
        series = prepped.series

        if len(series) < 30:
            raise ValueError("Insufficient data: pipeline requires at least 30 observations")

        # 3. Explore
        analyzer = EDAAnalyzer()
        eda = analyzer.analyze(series, frequency=freq).to_dict()

        # 4. Backtest
        backtester = Backtester(
            n_folds=cfg.backtest_folds,
            selection_metric=cfg.selection_metric,
            initial_train_frac=cfg.initial_train_frac,
        )
        models = create_models(
            series, frequency=freq, with_xgboost=cfg.include_xgboost
        )
        backtest = backtester.backtest(series, models=models, frequency=freq).to_dict()

        # 5. Select best model & retrain on full history
        best_name = backtest["best_model"]
        if best_name is None:
            raise ValueError("Backtesting failed: no model produced valid scores")

        best_model = next((m for m in models if m.name == best_name), None)
        if best_model is None:
            raise ValueError(f"Best model {best_name} not found in model set")

        best_model.fit(series, frequency=freq)
        fc = best_model.predict(cfg.horizon)

        forecast_payload = {
            "model": fc.model_name,
            "method": fc.method,
            "horizon": fc.horizon,
            "start_date": str(series.index[-1].date()),
            "forecast_dates": _future_dates(series.index[-1], fc.horizon, freq),
            "forecast": [float(x) for x in np.asarray(fc.forecast)],
            "lower": [float(x) for x in np.asarray(fc.lower)] if fc.lower is not None else None,
            "upper": [float(x) for x in np.asarray(fc.upper)] if fc.upper is not None else None,
            "historical": {
                "dates": [str(d.date()) for d in series.index],
                "values": [float(x) for x in series.values],
            },
            "metadata": fc.metadata,
            "expected_change": expected_change(fc.forecast, series.iloc[-1]),
        }

        # 6. Explainability
        explain = explain_model(best_model, fc, series, eda)

        return PipelineResult(
            report=report,
            eda=eda,
            backtest=backtest,
            forecast=forecast_payload,
            explainability=explain,
            config=cfg.to_dict(),
        )


def _future_dates(last_date: pd.Timestamp, horizon: int, freq: str) -> list[str]:
    freq_map = {"D": "D", "W": "W", "MS": "MS", "M": "MS", "QS": "QS", "Q": "QS", "YS": "YS", "A": "YS", "Y": "YS"}
    f = freq_map.get(freq, "D")
    try:
        future = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq=f)
        return [str(d.date()) for d in future]
    except Exception:
        return [str((last_date + pd.Timedelta(days=i + 1)).date()) for i in range(horizon)]


def expected_change(forecast: np.ndarray, last_actual: float) -> dict[str, Any]:
    """Expected change of the forecast relative to the last actual value."""
    fc = np.asarray(forecast, dtype=float)
    horizon = len(fc)
    target = fc[-1] if horizon > 0 else last_actual
    pct = (target - last_actual) / last_actual * 100 if last_actual != 0 else None
    return {
        "absolute": float(target - last_actual),
        "pct_change": round(pct, 4) if pct is not None else None,
        "reference": float(last_actual),
        "direction": "increase" if target > last_actual else "decrease" if target < last_actual else "flat",
        "period": f"last {horizon} periods",
    }


def explain_model(model, fc, series: pd.Series, eda: dict[str, Any]) -> dict[str, Any]:
    """Build an explainability payload for the selected model."""
    importance = []
    if hasattr(model, "feature_importance"):
        try:
            importance = model.feature_importance()
        except Exception:
            importance = []

    trend = eda.get("trend", {})
    seasonality = eda.get("seasonality", {})
    volatility = eda.get("volatility", {})

    drivers = []
    drivers.append(
        {
            "driver": "recent trend",
            "signal": trend.get("direction", "unknown"),
            "direction": "positive" if trend.get("direction") == "increasing" else ("negative" if trend.get("direction") == "decreasing" else "neutral"),
        }
    )
    if seasonality.get("detected"):
        drivers.append(
            {
                "driver": f"{seasonality.get('seasonality_type')} seasonality",
                "signal": f"period={seasonality.get('period')}, strength={seasonality.get('strength')}",
                "direction": "positive",
            }
        )
    drivers.append(
        {
            "driver": "recent volatility",
            "signal": volatility.get("trend", "unknown"),
            "direction": "negative" if volatility.get("trend") == "increasing" else "neutral",
        }
    )

    return {
        "model": model.name,
        "method": fc.method,
        "drivers": drivers,
        "feature_importance": importance[:15],
        "caveat": (
            "Signals describe statistical association with the forecast, "
            "not causal effects. No causal analysis was performed."
        ),
        "uncertainty_note": (
            "Prediction intervals widen with horizon; treat the midpoint as the "
            "expected value and the band as the plausible range."
        ),
    }