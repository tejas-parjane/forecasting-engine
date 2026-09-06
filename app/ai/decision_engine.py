"""AI decision-intelligence engine.

The engine compiles structured outputs from the forecasting pipeline into a
compact summary, then asks an LLM (if configured) to interpret it into a
business decision brief.

The system works entirely without the LLM: a rule-based fallback generates
a deterministic decision brief from the same structured summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.ai.prompts import build_full_report_prompt, FROM_STATISTICS
from app.exceptions import AIProviderError


@dataclass
class DecisionBrief:
    forecast_summary: str
    driver_analysis: str
    risk_assessment: str
    business_recommendation: str
    generated_by: str = "rule-based"

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_summary": self.forecast_summary,
            "driver_analysis": self.driver_analysis,
            "risk_assessment": self.risk_assessment,
            "business_recommendation": self.business_recommendation,
            "generated_by": self.generated_by,
        }


class DecisionEngine:
    """Compiles forecast pipeline output into an AI-interpreted decision brief."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 800,
        provider: str = "openai",
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self._client = None if not api_key else self._make_client()

    def _make_client(self):
        try:
            from openai import OpenAI

            return OpenAI(api_key=self.api_key)
        except Exception:
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def build_structured_input(
        self,
        forecast_payload: dict[str, Any],
        eda: dict[str, Any] | None = None,
        explainability: dict[str, Any] | None = None,
        report: dict[str, Any] | None = None,
        backtest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile a compact, LLM-safe summary of the pipeline output."""
        fc = forecast_payload
        forecast_values = fc.get("forecast", [])
        lower = fc.get("lower") or []
        upper = fc.get("upper") or []

        def mean(xs):
            return round(sum(xs) / len(xs), 2) if xs else None

        summary = {
            "forecast": {
                "model": fc.get("model"),
                "horizon": fc.get("horizon"),
                "horizon_unit": "periods",
                "start_date": fc.get("start_date"),
                "forecast_end_value": forecast_values[-1] if forecast_values else None,
                "horizon_mean": mean(forecast_values),
                "expected_change_pct": fc.get("expected_change", {}).get("pct_change"),
                "expected_change_absolute": fc.get("expected_change", {}).get("absolute"),
                "direction": fc.get("expected_change", {}).get("direction"),
            },
            "uncertainty": {
                "lower_mean": mean(lower),
                "upper_mean": mean(upper),
                "lower_end": lower[-1] if lower else None,
                "upper_end": upper[-1] if upper else None,
                "method": fc.get("metadata", {}).get("uncertainty_method"),
            },
            "trend": (eda or {}).get("trend"),
            "seasonality": (eda or {}).get("seasonality"),
            "volatility": (eda or {}).get("volatility"),
            "anomaly_count": (eda or {}).get("anomalies", {}).get("count"),
            "top_drivers": (explainability or {}).get("drivers", [])[:6],
            "model_metrics": self._best_model_metrics(backtest, fc.get("model")),
            "data_quality": {
                "score": (report or {}).get("quality_score"),
                "missing_values": (report or {}).get("missing_values"),
                "outlier_count": (report or {}).get("outlier_count"),
                "note": (
                    (report or {}).get("issues", [{}])[0].get("message", "")
                    if (report or {}).get("issues")
                    else "No known issues"
                ),
            },
        }
        return summary

    def _best_model_metrics(self, backtest: dict[str, Any] | None, model_name: str | None) -> dict[str, Any] | None:
        if not backtest:
            return None
        for m in backtest.get("models", []):
            if m.get("model_name") == model_name:
                agg = m.get("aggregated", {})
                return {
                    metric: agg.get(metric)
                    for metric in ("mae", "rmse", "mape", "smape", "wape")
                }
        return None

    def generate(self, structured_input: dict[str, Any], use_llm: bool | None = None) -> DecisionBrief:
        if use_llm is None:
            use_llm = self.enabled
        if use_llm and self.enabled:
            return self._generate_with_llm(structured_input)
        return self._generate_rule_based(structured_input)

    def _generate_with_llm(self, structured_input: dict[str, Any]) -> DecisionBrief:
        if self._client is None:
            raise AIProviderError("AI provider not configured")

        prompt = build_full_report_prompt(structured_input)
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": FROM_STATISTICS},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            return self._parse_sections(text, generated_by=f"llm:{self.model}")
        except Exception as exc:
            raise AIProviderError(f"LLM generation failed: {exc}") from exc

    def _parse_sections(self, text: str, generated_by: str) -> DecisionBrief:
        def extract(heading: str) -> str:
            remaining = text
            lower_heading = heading.lower()
            pos = remaining.lower().find(lower_heading)
            if pos == -1:
                return ""
            # Return from the heading to the next ## heading or end
            body = remaining[pos:].split("##")[1] if len(remaining[pos:].split("##")) > 1 else remaining[pos:]
            if body.startswith("#"):
                body = body.lstrip("#").strip()
            # Find next heading within body
            parts = body.split("## ")
            return parts[0].strip() if parts else ""

        forecast_summary = extract("Forecast Summary") or extract("summary")
        driver_analysis = extract("Driver Analysis") or extract("drivers")
        risk_assessment = extract("Risk Assessment") or extract("risk")
        recommendation = extract("Business Recommendation") or extract("recommendation")

        # Fallback: if anything is empty, use the whole text
        if not forecast_summary and not recommendation:
            return DecisionBrief(
                forecast_summary=text.strip(),
                driver_analysis="",
                risk_assessment="",
                business_recommendation="",
                generated_by=generated_by,
            )
        return DecisionBrief(
            forecast_summary=forecast_summary,
            driver_analysis=driver_analysis,
            risk_assessment=risk_assessment,
            business_recommendation=recommendation,
            generated_by=generated_by,
        )

    def _generate_rule_based(self, structured_input: dict[str, Any]) -> DecisionBrief:
        fc = structured_input["forecast"]
        unc = structured_input["uncertainty"]
        trend = structured_input.get("trend", {}) or {}
        seas = structured_input.get("seasonality", {}) or {}
        vol = structured_input.get("volatility", {}) or {}
        drivers = structured_input.get("top_drivers", []) or []

        direction = fc.get("direction", "flat")
        pct = fc.get("expected_change_pct")
        horizon = fc.get("horizon", 0)
        model = fc.get("model", "unknown")

        pct_str = f"{pct:.1f}%" if pct is not None else "not available"
        direction_word = {"increase": "increase", "decrease": "decrease", "flat": "remain stable"}.get(direction, "change")

        summary = (
            f"Over the next {horizon} period(s), demand is expected to {direction_word} "
            f"by approximately {pct_str}. This forecast is produced by the {model} model "
            "and reflects the recent trend and seasonal pattern observed in the data."
        )

        if drivers:
            pos = [d for d in drivers if d.get("direction") == "positive"]
            neg = [d for d in drivers if d.get("direction") == "negative"]
            pos_s = ", ".join(d["signal"] for d in pos) if pos else "no dominant positive signals"
            neg_s = ", ".join(d["signal"] for d in neg) if neg else "no significant negative signals"
            driver_text = (
                f"Primary positive signals: {pos_s}. "
                f"Primary negative signals: {neg_s}. "
                "These describe statistical association, not causation."
            )
        else:
            driver_text = (
                "The forecast is driven primarily by the continuation of recent values and seasonality. "
                "No causal relationship is implied."
            )

        lower_mean = unc.get("lower_mean")
        upper_mean = unc.get("upper_mean")
        uncertainty_txt = "not quantified"
        if lower_mean is not None and upper_mean is not None:
            uncertainty_txt = f"approximately {lower_mean:.0f} to {upper_mean:.0f} across the horizon"

        risk = (
            f"Forecast uncertainty spans {uncertainty_txt}. "
            f"Recent volatility trend: {vol.get('trend', 'unknown')}. "
        )
        if seas.get("detected"):
            risk += f"A {seas.get('seasonality_type', 'seasonal')} pattern (strength {seas.get('strength')}) is present; "
        risk += "Treat the interval midpoint as the expected value and the bounds as a plausible planning range."

        recommendation = self._build_recommendation(direction, uncertainty_txt, pct_str, lower_mean, upper_mean)

        return DecisionBrief(
            forecast_summary=summary,
            driver_analysis=driver_text,
            risk_assessment=risk,
            business_recommendation=recommendation,
            generated_by="rule-based",
        )

    def _build_recommendation(self, direction: str, uncertainty_txt: str, pct_str: str, lower: float | None, upper: float | None) -> str:
        if direction == "increase":
            rec = (
                f"Plan for rising demand: the projection indicates an increase of {pct_str}. "
                "Consider scaling capacity, inventory, or staffing incrementally if you can act "
                "in stages, and review upstream capacity constraints."
            )
        elif direction == "decrease":
            rec = (
                f"Expect softer demand: the projection indicates a decrease of {pct_str}. "
                "Avoid over-committing inventory or staffing now; plan a conservative restocking "
                "and cost-tightening posture."
            )
        else:
            rec = (
                "Demand is expected to remain broadly stable. Maintain current capacity and "
                "watch for a breakout outside the forecast band before changing plans."
            )

        if lower is not None and upper is not None:
            rec += (
                f" Given the plausible range of approximately {lower:.0f} to {upper:.0f}, "
                "prefer planning toward the lower-middle of the band unless additional "
                "confirming signals appear."
            )
        rec += " Forecasts are estimates; revisit the plan as new data arrives."
        return rec