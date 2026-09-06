"""Time-series data validation.

Produces a structured DataQualityReport describing schema issues,
temporal integrity, and target quality without silently modifying data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # 'error' | 'warning' | 'info'
    category: str  # 'schema' | 'temporal' | 'target' | 'duplicate'
    message: str


@dataclass
class DataQualityReport:
    """Structured validation output."""

    is_valid: bool
    quality_score: float  # 0-100
    missing_values: int
    duplicate_records: int
    date_coverage_days: int
    detected_frequency: str
    n_observations: int
    outlier_count: int
    target_constant: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    transformations_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "quality_score": round(self.quality_score, 1),
            "missing_values": self.missing_values,
            "duplicate_records": self.duplicate_records,
            "date_coverage_days": self.date_coverage_days,
            "detected_frequency": self.detected_frequency,
            "n_observations": self.n_observations,
            "outlier_count": self.outlier_count,
            "target_constant": self.target_constant,
            "issues": [
                {"severity": i.severity, "category": i.category, "message": i.message}
                for i in self.issues
            ],
            "transformations_applied": self.transformations_applied,
        }


def detect_frequency(index: pd.DatetimeIndex) -> str:
    """Detect the most likely regular frequency of a DatetimeIndex.

    Returns a pandas-style frequency label: 'D', 'W', 'MS', 'QS', 'YS', or 'IR' (irregular).
    """
    if index is None or len(index) < 2:
        return "IR"
    deltas = np.diff(index.values.astype("int64") // 10**9)
    if len(deltas) == 0:
        return "IR"
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return "IR"
    median_delta = float(np.median(deltas))
    tolerance = 0.15

    day = 86400
    if abs(median_delta - day) / day < tolerance:
        return "D"
    if abs(median_delta - 7 * day) / (7 * day) < tolerance:
        return "W"
    if abs(median_delta - 30 * day) / (30 * day) < tolerance:
        return "MS"
    if abs(median_delta - 30.5 * day) / (30.5 * day) < tolerance:
        return "MS"
    if abs(median_delta - 90 * day) / (90 * day) < tolerance:
        return "QS"
    if abs(median_delta - 365 * day) / (365 * day) < tolerance:
        return "YS"
    return "IR"


def _iqr_outliers(values: pd.Series) -> pd.Series:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=values.index)
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    return (values < lower) | (values > upper)


class DataValidator:
    """Validates a time-series dataset and produces a DataQualityReport."""

    def __init__(
        self,
        date_col: str = "date",
        target_col: str = "target",
        allow_negative: bool = True,
        outlier_iqr_multiple: float = 3.0,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.allow_negative = allow_negative
        self.outlier_iqr_multiple = outlier_iqr_multiple

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        issues: list[ValidationIssue] = []
        transformations: list[str] = []
        work = df.copy()

        # ---- Schema checks ----
        if self.date_col not in work.columns:
            issues.append(ValidationIssue("error", "schema", f"Date column '{self.date_col}' not found."))
        if self.target_col not in work.columns:
            issues.append(ValidationIssue("error", "schema", f"Target column '{self.target_col}' not found."))

        if self.date_col not in work.columns or self.target_col not in work.columns:
            return DataQualityReport(
                is_valid=False, quality_score=0.0, missing_values=0, duplicate_records=0,
                date_coverage_days=0, detected_frequency="IR", n_observations=len(work),
                outlier_count=0, target_constant=False, issues=issues,
            )

        # ---- Date parsing ----
        try:
            work[self.date_col] = pd.to_datetime(work[self.date_col], errors="coerce")
            n_bad_dates = int(work[self.date_col].isna().sum())
            if n_bad_dates > 0:
                issues.append(ValidationIssue("error", "schema", f"{n_bad_dates} rows have invalid dates."))
                work = work[work[self.date_col].notna()]
            work = work.sort_values(self.date_col).reset_index(drop=True)
            transformations.append("Parsed date column and sorted chronologically")
        except Exception as exc:
            issues.append(ValidationIssue("error", "schema", f"Failed to parse dates: {exc}"))
            return DataQualityReport(
                is_valid=False, quality_score=0.0, missing_values=0, duplicate_records=0,
                date_coverage_days=0, detected_frequency="IR", n_observations=len(work),
                outlier_count=0, target_constant=False, issues=issues,
            )

        # ---- Duplicate dates ----
        duplicate_dates = int(work[self.date_col].duplicated().sum())
        if duplicate_dates > 0:
            issues.append(
                ValidationIssue("warning", "duplicate", f"{duplicate_dates} duplicate date records found.")
            )
        duplicate_records = int(work.duplicated().sum())

        # ---- Missing values ----
        missing = int(work[self.target_col].isna().sum())
        if missing > 0:
            issues.append(
                ValidationIssue("warning", "target", f"{missing} missing values in target column.")
            )

        # ---- Target numeric coercion ----
        try:
            work[self.target_col] = pd.to_numeric(work[self.target_col], errors="coerce")
            work = work[work[self.target_col].notna()].reset_index(drop=True)
            if missing > 0:
                transformations.append(f"Dropped {missing} rows with missing target")
        except Exception as exc:
            issues.append(ValidationIssue("error", "target", f"Target column not numeric: {exc}"))

        if len(work) == 0:
            return DataQualityReport(
                is_valid=False, quality_score=0.0, missing_values=missing,
                duplicate_records=duplicate_dates, date_coverage_days=0,
                detected_frequency="IR", n_observations=0, outlier_count=0,
                target_constant=False, issues=issues, transformations_applied=transformations,
            )

        # ---- Frequency detection ----
        freq = detect_frequency(work[self.date_col])

        # ---- Date coverage / gaps ----
        if len(work) >= 2:
            date_range = (work[self.date_col].max() - work[self.date_col].min()).days + 1
        else:
            date_range = 1

        expected_count = date_range if freq in ("D",) else len(work)
        gap_likely = freq == "D" and len(work) < date_range * 0.95
        if gap_likely:
            n_missing_dates = date_range - len(work)
            issues.append(
                ValidationIssue("warning", "temporal", f"Likely {n_missing_dates} missing daily date(s) (coverage {len(work)}/{date_range} days).")
            )

        if freq == "IR":
            issues.append(
                ValidationIssue("warning", "temporal", "Date frequency appears irregular or inconsistent.")
            )

        # ---- Negative values ----
        negatives = int((work[self.target_col] < 0).sum())
        if negatives > 0 and not self.allow_negative:
            issues.append(
                ValidationIssue("error", "target", f"{negatives} negative values found but negatives disallowed.")
            )

        # ---- Outliers ----
        outliers = _iqr_outliers(work[self.target_col])
        outlier_count = int(outliers.sum())
        if outlier_count > 0:
            issues.append(
                ValidationIssue("info", "target", f"{outlier_count} extreme outliers detected (IQR x {self.outlier_iqr_multiple}).")
            )

        # ---- Constant series ----
        target_constant = bool(work[self.target_col].nunique() <= 1)

        # ---- Quality score ----
        deductions = 0.0
        if missing > 0:
            deductions += 10
        if duplicate_dates > 0:
            deductions += 10
        if gap_likely:
            deductions += 10
        if freq == "IR":
            deductions += 5
        if outlier_count > 0:
            deductions += 5
        if target_constant:
            deductions += 20
        has_errors = any(i.severity == "error" for i in issues)
        quality_score = max(0.0, 100.0 - deductions)
        if has_errors:
            quality_score = min(quality_score, 50.0)

        return DataQualityReport(
            is_valid=not has_errors,
            quality_score=quality_score,
            missing_values=missing,
            duplicate_records=duplicate_dates,
            date_coverage_days=date_range,
            detected_frequency=freq,
            n_observations=len(work),
            outlier_count=outlier_count,
            target_constant=target_constant,
            issues=issues,
            transformations_applied=transformations,
        )
