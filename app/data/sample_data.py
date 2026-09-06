"""Generate a synthetic business demand time series for demonstration.

The dataset exhibits:
  - a clear upward trend
  - weekly + yearly seasonality
  - random noise
  - occasional anomalies
  - some promotional / holiday spikes

All randomness is seeded for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_business_demand(
    n_days: int = 1095,
    start: str = "2021-01-01",
    seed: int = 42,
    base: float = 800.0,
    trend_per_year: float = 120.0,
    noise_scale: float = 55.0,
    anomaly_prob: float = 0.015,
) -> pd.DataFrame:
    """Return a daily business-demand DataFrame with columns [date, target].

    Args:
        n_days: Number of daily observations (default 1095 ≈ 3 years).
        start: First date.
        seed: Reproducible RNG seed.
        base: Baseline demand level.
        trend_per_year: Linear increase in demand per year.
        noise_scale: Std dev of random noise.
        anomaly_prob: Probability a day is an anomalous spike/dip.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    t = np.arange(n_days)

    # Linear trend
    trend = base + trend_per_year * (t / 365.0)

    # Yearly seasonality (sinusoidal)
    yearly = 140.0 * np.sin(2 * np.pi * t / 365.0 + 1.0)

    # Weekly seasonality (weekday effect)
    weekday = dates.dayofweek.values
    weekly_effect = np.where(
        weekday < 5,
        40.0 * np.sin(weekday / 5 * np.pi) + 30.0,
        -120.0,
    )

    # Monthly seasonality (mild)
    monthly = 25.0 * np.sin(2 * np.pi * (dates.day.values - 1) / 30.0)

    # Holiday / promotional spikes at year end (black friday-like)
    promo = np.where((dates.month == 11) & ((dates.day.values >= 20) | (dates.day.values <= 30)), 180.0, 0.0)

    # Noise
    noise = rng.normal(0, noise_scale, n_days)

    target = trend + yearly + weekly_effect + monthly + promo + noise
    target = np.maximum(target, 100.0)

    # Inject anomalies: occasional sharp spikes
    anomaly_mask = rng.random(n_days) < anomaly_prob
    anomaly_size = rng.choice([-1, 1], size=anomaly_mask.sum()) * rng.uniform(250.0, 500.0, size=anomaly_mask.sum())
    target[anomaly_mask] += anomaly_size
    target = np.maximum(target, 100.0)

    df = pd.DataFrame({"date": dates, "target": np.round(target, 2)})
    return df


def save_sample_data(path: str | Path, **kwargs) -> Path:
    """Generate and save the sample dataset to disk."""
    path = Path(path)
    df = generate_business_demand(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "sample" / "business_demand.csv"
    saved = save_sample_data(out)
    print(f"Sample dataset written to {saved} ({len(pd.read_csv(saved))} rows)")
