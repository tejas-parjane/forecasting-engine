"""Time-series backtesting via expanding-window validation.

No random train/test splits. Each fold trains on a growing window of the
past and evaluates on a fixed-size validation window that immediately
follows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.data.validator import detect_frequency
from app.evaluation.metrics import compute_metrics
from app.models.base import BaseForecaster
from app.models.registry import create_models


@dataclass
class ModelEvaluation:
    """Validation results for a single model across backtest folds."""

    model_name: str
    scores_per_fold: list[dict[str, Any]]
    aggregated: dict[str, Any]
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "scores_per_fold": self.scores_per_fold,
            "aggregated": self.aggregated,
            "rank": self.rank,
        }


@dataclass
class BacktestResult:
    """Full backtest output."""

    folds: list[dict[str, Any]]
    models: list[ModelEvaluation]
    selection_metric: str
    best_model: str | None = None
    best_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": self.folds,
            "models": [m.to_dict() for m in self.models],
            "selection_metric": self.selection_metric,
            "best_model": self.best_model,
            "best_score": self.best_score,
        }


class Backtester:
    """Runs expanding-window backtests over a candidate model set."""

    def __init__(
        self,
        n_folds: int = 4,
        selection_metric: str = "smape",
        initial_train_frac: float = 0.6,
        fold_growth: int = 0,
    ):
        self.n_folds = n_folds
        self.selection_metric = selection_metric
        self.initial_train_frac = initial_train_frac
        self.fold_growth = fold_growth

    def backtest(
        self,
        series: pd.Series,
        models: list[BaseForecaster] | None = None,
        frequency: str | None = None,
        horizon: int | None = None,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            series: Chronological target series.
            models: Models to evaluate. If None, all registered models are built.
            frequency: Detected data frequency.
            horizon: Fixed evaluation horizon per fold (default: derived from folds).
        """
        freq = frequency or detect_frequency(series.index)
        if models is None:
            models = create_models(series, frequency=freq)

        n = len(series)
        min_train = max(30, int(n * self.initial_train_frac))
        remaining = n - min_train
        if remaining < 10:
            raise ValueError(
                "Not enough data for backtesting. Reduce initial_train_frac or folds."
            )

        # Build fold boundaries. `remaining` shrinks each round and also feeds
        # the size guess, so the folds distribute across the available tail and
        # no (tr_end, va_end) can overshoot the end of the series.
        fold_sizes = []
        for i in range(self.n_folds):
            want = min(horizon or size_guess(remaining, self.n_folds - i, freq), remaining)
            size = min(max(7, want), remaining)
            fold_sizes.append(size)
            remaining -= size
        fold_sizes = [s for s in fold_sizes if s > 0]

        boundaries = []
        start = min_train
        for size in fold_sizes:
            boundaries.append((start, start + size))
            start += size

        folds_meta = []
        model_evaluations: list[ModelEvaluation] = []

        for model in models:
            per_fold = []
            for i, (tr_end, va_end) in enumerate(boundaries):
                train = series.iloc[:tr_end]
                val_actual = series.iloc[tr_end:va_end].values
                h = len(val_actual)

                try:
                    model.fit(train, frequency=freq)
                    result = model.predict(h)
                    fc = np.asarray(result.forecast)
                    if len(fc) < h:
                        fc = np.pad(fc, (0, h - len(fc)), mode="edge")
                    elif len(fc) > h:
                        fc = fc[:h]
                    metrics = compute_metrics(val_actual, fc)
                except Exception as exc:
                    metrics = {"error": str(exc), "n": h}
                    fc = np.full(h, np.nan)

                per_fold.append(
                    {
                        "fold": i + 1,
                        "train_start": str(train.index[0].date()),
                        "train_end": str(train.index[-1].date()),
                        "val_start": str(series.index[tr_end].date()),
                        "val_end": str(series.index[va_end - 1].date()),
                        "horizon": h,
                        "metrics": metrics,
                        "forecast": [float(x) for x in fc] if len(fc) else None,
                        "actual": [float(x) for x in val_actual],
                    }
                )

            aggregated = self._aggregate(per_fold)
            if self.selection_metric in aggregated and aggregated[self.selection_metric] is not None:
                aggregated["rank"] = None
            model_evaluations.append(ModelEvaluation(
                model_name=model.name,
                scores_per_fold=per_fold,
                aggregated=aggregated,
            ))

        # Rank by selection metric (lower is better)
        ranked = [m for m in model_evaluations if m.aggregated.get(self.selection_metric) is not None]
        ranked.sort(key=lambda m: m.aggregated[self.selection_metric])
        for i, m in enumerate(ranked):
            m.rank = i + 1

        best_model = ranked[0].model_name if ranked else None
        best_score = ranked[0].aggregated[self.selection_metric] if ranked else None

        return BacktestResult(
            folds=[{"fold": i + 1, "train": {"start": str(series.index[i].date())}} for i in range(len(boundaries))],
            models=model_evaluations,
            selection_metric=self.selection_metric,
            best_model=best_model,
            best_score=best_score,
        )

    def _aggregate(self, per_fold: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate metrics across folds, weighted by horizon length."""
        agg: dict[str, Any] = {}
        metric_keys = ["mae", "rmse", "mape", "smape", "wape"]

        for key in metric_keys:
            vals = []
            weights = []
            for f in per_fold:
                m = f.get("metrics", {})
                if key in m and m[key] is not None and not isinstance(m[key], str):
                    vals.append(m[key])
                    weights.append(f["horizon"])
            if vals:
                if key == "mape":
                    # Only aggregate where defined across all folds consistently
                    agg[key] = round(float(np.mean(vals)), 4)
                else:
                    agg[key] = round(float(np.average(vals, weights=weights)), 4)
            else:
                agg[key] = None

        # Count successes
        agg["n_folds_valid"] = sum(
            1 for f in per_fold if "error" not in f.get("metrics", {}) and f.get("metrics", {}).get("smape") is not None
        )
        agg["n_folds_total"] = len(per_fold)
        return agg


def size_guess(available: int, folds_left: int, freq: str) -> int:
    """Heuristic for fold size."""
    default = {"D": 14, "W": 4, "MS": 3, "QS": 2, "YS": 1}.get(freq, 5)
    if folds_left <= 0:
        return default
    return min(max(default, available // folds_left), available)