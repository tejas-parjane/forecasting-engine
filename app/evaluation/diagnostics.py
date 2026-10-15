"""In-sample residual diagnostics for fitted forecasters.

Answers: are the model's leftovers white noise, or is there predictable
structure the model failed to capture? Uses the Ljung-Box test (no
autocorrelation up to lag ``lags``) and the Jarque-Bera test (normality),
plus basic moments. Pure function — no model dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def residual_diagnostics(residuals: np.ndarray | list[float], lags: int = 10) -> dict[str, Any]:
    """Summarize in-sample residuals and test whiteness/normality.

    Returns a JSON-serializable dict with ``n``, ``mean``, ``std``,
    ``skew``, ``kurtosis`` (excess), ``ljung_box`` (statistic + p-value at
    ``lags``) and ``jarque_bera`` (statistic + p-value). ``None`` p-values
    mean the test could not run (e.g. constant residuals).
    """
    from scipy import stats

    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    out: dict[str, Any] = {"n": int(len(r))}
    if len(r) < max(lags + 2, 4):
        out.update({"mean": None, "std": None, "skew": None, "kurtosis": None,
                    "ljung_box": None, "jarque_bera": None,
                    "note": "too few residuals for diagnostics"})
        return out

    out["mean"] = round(float(np.mean(r)), 4)
    out["std"] = round(float(np.std(r, ddof=1)), 4)
    out["skew"] = round(float(stats.skew(r)), 4)
    out["kurtosis"] = round(float(stats.kurtosis(r)), 4)  # excess kurtosis

    try:
        from statsmodels.stats.stattools import durbin_watson

        out["durbin_watson"] = round(float(durbin_watson(r)), 4)
    except Exception:
        out["durbin_watson"] = None

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(r, lags=[lags], return_df=True).iloc[0]
        out["ljung_box"] = {
            "lags": lags,
            "statistic": round(float(lb["lb_stat"]), 4),
            "p_value": round(float(lb["lb_pvalue"]), 4),
            "white_noise": bool(lb["lb_pvalue"] > 0.05),
        }
    except Exception:
        out["ljung_box"] = None

    try:
        jb_stat, jb_p = stats.jarque_bera(r)
        out["jarque_bera"] = {
            "statistic": round(float(jb_stat), 4),
            "p_value": round(float(jb_p), 4),
            "normal": bool(jb_p > 0.05),
        }
    except Exception:
        out["jarque_bera"] = None

    return out
