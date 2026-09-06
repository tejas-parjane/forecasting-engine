"""Forecast explainability: drivers, feature importance, and SHAP where feasible."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Explanation:
    model_name: str
    drivers: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    shap_values: dict[str, Any] | None = None
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "drivers": self.drivers,
            "feature_importance": self.feature_importance,
            "shap_summary": self.shap_values,
            "caveats": self.caveats,
        }


class ExplainabilityAnalyzer:
    """Produces structured explanations for a trained forecast model."""

    def __init__(self, use_shap: bool = False):
        self.use_shap = use_shap

    def explain(
        self,
        model,
        series: pd.Series,
        frequency: str | None = None,
        trend_dict: dict[str, Any] | None = None,
        seasonality_dict: dict[str, Any] | None = None,
        volatility_dict: dict[str, Any] | None = None,
    ) -> Explanation:
        drivers = []
        caveats = [
            "Signals describe statistical association, not causation. No causal analysis was performed."
        ]

        trend = trend_dict or {}
        seasonality = seasonality_dict or {}
        volatility = volatility_dict or {}

        if trend.get("direction"):
            if trend["direction"] == "increasing":
                drivers.append({"driver": "recent trend", "signal": "increasing", "direction": "positive"})
            elif trend["direction"] == "decreasing":
                drivers.append({"driver": "recent trend", "signal": "decreasing", "direction": "negative"})
            else:
                drivers.append({"driver": "recent trend", "signal": "stable", "direction": "neutral"})

        if seasonality.get("detected"):
            drivers.append(
                {
                    "driver": f"{seasonality.get('seasonality_type')} seasonality",
                    "signal": f"period={seasonality.get('period')}, strength={round(seasonality.get('strength', 0), 3)}",
                    "direction": "positive" if seasonality.get("strength", 0) > 0.3 else "neutral",
                }
            )

        if volatility.get("trend"):
            vt = volatility["trend"]
            drivers.append(
                {
                    "driver": "recent volatility",
                    "signal": vt,
                    "direction": "negative" if vt == "increasing" else "neutral",
                }
            )

        drivers.append(
            {
                "driver": "recent value",
                "signal": f"last observed = {float(series.iloc[-1]):.2f}",
                "direction": "neutral",
            }
        )

        importance = []
        shap_result = None
        if hasattr(model, "feature_importance"):
            try:
                importance = model.feature_importance()
            except Exception:
                importance = []

        if self.use_shap and importance:
            try:
                shap_result = self._shap_summary(model)
            except Exception:
                shap_result = None
            if shap_result is None:
                caveats.append("SHAP values unavailable; feature importance used instead.")

        return Explanation(
            model_name=getattr(model, "name", "unknown"),
            drivers=drivers,
            feature_importance=importance,
            shap_values=shap_result,
            caveats=caveats,
        )

    def _shap_summary(self, model) -> dict[str, Any] | None:
        try:
            import shap

            if not hasattr(model, "model_") or getattr(model, "model_", None) is None:
                return None
            if not hasattr(model, "feature_cols_") or not hasattr(model, "last_history_"):
                return None

            feats = model.feature_cols_
            hist = model.last_history_full_
            X = hist[feats].astype(float).iloc[-min(100, len(hist)):]

            explainer = shap.TreeExplainer(model.model_)
            shaps = explainer.shap_values(X)
            if isinstance(shaps, list):
                shaps = np.array(shaps[-1] if len(shaps) else 0)
            mean_abs = np.mean(np.abs(np.asarray(shaps)), axis=0)
            pairs = sorted(zip(feats, mean_abs), key=lambda x: -x[1])
            return {
                "method": "TreeExplainer (mean |SHAP|)",
                "top_features": [
                    {"feature": str(f), "mean_abs_shap": round(float(v), 4)} for f, v in pairs[:10]
                ],
            }
        except Exception:
            return None