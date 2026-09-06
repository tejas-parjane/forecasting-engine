"""XGBoost regression forecasting with recursive multi-step prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.data.validator import detect_frequency
from app.features.time_features import calendar_features, default_lags, lag_features, rolling_features
from app.models.base import BaseForecaster, ForecastResult


class XGBoostForecaster(BaseForecaster):
    """ML forecasting using XGBoost on engineered time features.

    Strategy:
      - Engineer calendar + lag + rolling features (past-only).
      - Train with a warm-up drop on the earliest rows (whose lag features
        are NaN) to avoid leakage.
      - Predict recursively: at each future step the newly predicted value
        is fed back as the latest "observed" value for subsequent lags.

    Uncertainty: quantile regression. During backtesting we collect residual
    quantiles; in forecast generation we assume residuals ~ historical
    distribution and build an empirical prediction interval by perturbing
    stored tree leaf predictions is not available, so we approximate with a
    Gaussian band scaled by the square-root-of-horizon rule.
    """

    name = "xgboost"

    def __init__(self, horizon_adjust: bool = False, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 5, random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_: Any = None
        self.frequency_ = "D"
        self.feature_cols_: list[str] = []
        self.mean_y_ = 0.0
        self.std_y_ = 0.0
        self.residual_std_ = 0.0
        self.series_: pd.Series | None = None

    def fit(self, series: pd.Series, frequency: str = "D", **kwargs: Any) -> "XGBoostForecaster":
        import xgboost as xgb

        self.frequency_ = frequency
        df = pd.DataFrame({"date": series.index, "target": series.values})

        # Frequency-based lag/rolling configuration
        lags = default_lags(frequency)
        windows = {
            "D": [7, 14, 28],
            "W": [4, 13, 26],
            "MS": [3, 6, 12],
            "QS": [2, 4],
        }.get(frequency, [7, 28])

        frame = calendar_features(df)
        frame = lag_features(frame, "target", lags=lags, frequency=frequency)
        frame = rolling_features(frame, "target", windows=windows, frequency=frequency)

        # Drop any feature column that ended up entirely NaN (e.g. a yearly lag
        # on short history), then drop warm-up rows whose features are
        # incomplete. Dropping the first k rows is safe: those rows simply lack
        # history and never leak future information.
        frame = frame.dropna(axis=1, how="all")
        feature_cols = [c for c in frame.columns if c not in ("date", "target")]
        frame = frame.dropna().reset_index(drop=True)
        if len(frame) < 30:
            raise ValueError("Insufficient data after feature engineering for XGBoost")

        X = frame[feature_cols].astype(float)
        y = frame["target"].astype(float)

        # Standardize target for stability
        self.mean_y_ = float(y.mean())
        self.std_y_ = float(y.std())
        if self.std_y_ == 0:
            self.std_y_ = 1.0
        y_std = (y - self.mean_y_) / self.std_y_

        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=self.random_state,
        )
        self.model_.fit(X, y_std)

        # Residual std in standardized space
        preds_std = self.model_.predict(X)
        resid = y_std - preds_std
        self.residual_std_ = float(np.std(resid))

        self.feature_cols_ = feature_cols
        self.series_ = series.copy()
        return self

    def predict(self, horizon: int, **kwargs: Any) -> ForecastResult:
        if self.model_ is None:
            raise RuntimeError("Model not fitted")

        horizon = int(horizon)
        freq = self.frequency_
        lags = default_lags(freq)
        windows = {
            "D": [7, 14, 28],
            "W": [4, 13, 26],
            "MS": [3, 6, 12],
            "QS": [2, 4],
        }.get(freq, [7, 28])

        # Chronological frame of known/predicted values. Features computed on the
        # last row (whose target is known) describe what is knowable at that point
        # in time, so they are used to predict the NEXT future step. This is the
        # standard recursive construction and leaks nothing about the future.
        hist = pd.DataFrame({"date": self.series_.index, "target": self.series_.values})
        future_dates = pd.date_range(
            start=self.series_.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        predicted: list[float] = []
        for i in range(horizon):
            cal = calendar_features(hist)
            cal = lag_features(cal, "target", lags=lags, frequency=freq)
            cal = rolling_features(cal, "target", windows=windows, frequency=freq)

            latest = cal.iloc[-1]
            feats = latest[self.feature_cols_].astype(float)
            X_row = feats.values.reshape(1, -1)
            pred_std = float(self.model_.predict(X_row)[0])
            pred = pred_std * self.std_y_ + self.mean_y_

            predicted.append(pred)
            hist = pd.concat(
                [hist, pd.DataFrame({"date": [future_dates[i]], "target": [pred]})],
                ignore_index=True,
            )

        fc = np.array(predicted)

        # Uncertainty band: assume residuals ~ N(0, residual_std * std_y)
        # scaled by sqrt(t) growth.
        z = 1.96
        scale = self.residual_std_ * self.std_y_
        sqrt_scale = np.sqrt(np.arange(1, horizon + 1))
        lower = fc - z * scale * sqrt_scale
        upper = fc + z * scale * sqrt_scale

        return ForecastResult(
            model_name=self.name,
            horizon=horizon,
            forecast=fc,
            lower=lower,
            upper=upper,
            method="xgboost recursive (past-only features)",
            metadata={
                "n_features": len(self.feature_cols_),
                "n_estimators": self.n_estimators,
                "residual_std": round(self.residual_std_ * self.std_y_, 4),
                "uncertainty_method": "gaussian band scaled by sqrt(horizon)",
            },
        )

    def feature_importance(self) -> list[dict[str, Any]]:
        """Return feature importances as a sorted list of dicts."""
        if self.model_ is None:
            return []
        imp = self.model_.feature_importances_
        pairs = sorted(zip(self.feature_cols_, imp), key=lambda x: -x[1])
        return [{"feature": str(f), "importance": float(i)} for f, i in pairs]