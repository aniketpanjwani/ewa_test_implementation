"""Prediction utilities for next-payday estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from risk_model.datasets import RawDataset, load_raw_dataset


TimestampLike = Union[str, datetime, pd.Timestamp, None]


@dataclass
class PredictionResult:
    """Structured response for next payday prediction."""

    date: pd.Timestamp
    low: Optional[pd.Timestamp] = None
    high: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        def _fmt(ts: Optional[pd.Timestamp]) -> Optional[str]:
            return ts.strftime("%Y-%m-%d") if ts is not None else None

        result = {"date": _fmt(self.date)}
        if self.low is not None:
            result["low"] = _fmt(self.low)
        if self.high is not None:
            result["high"] = _fmt(self.high)
        return result


def _prepare_payroll(raw: RawDataset) -> pd.DataFrame:
    """Return payroll-only transactions as naive UTC timestamps."""

    payroll = raw.transactions[raw.transactions["category"] == "payroll"].copy()
    if payroll.empty:
        return pd.DataFrame(columns=["user_id", "posted_at"])
    payroll["posted_at"] = payroll["posted_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    payroll.sort_values(["user_id", "posted_at"], inplace=True, ignore_index=True)
    return payroll[["user_id", "posted_at"]]


def _month_last_day(ts: pd.Timestamp) -> int:
    # pandas monthrange: use period.
    return ts.days_in_month


def _set_day(ts: pd.Timestamp, target_day: int) -> pd.Timestamp:
    day = min(max(target_day, 1), _month_last_day(ts))
    return ts.replace(day=day)


def _adjust_weekend(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.weekday() == 5:
        return ts - pd.Timedelta(days=1)
    if ts.weekday() == 6:
        return ts + pd.Timedelta(days=1)
    return ts


def _parse_anchors(value: str) -> List[int]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    anchors = []
    for part in parts:
        try:
            anchors.append(int(part))
        except ValueError:
            continue
    return sorted(set(anchors))


def _series_to_numpy_days(series: pd.Series) -> np.ndarray:
    return series.values.astype("datetime64[D]")


class PaydatePredictor:
    """Predict next payroll dates using cadence heuristics plus residual bias."""

    def __init__(self, data_dir: Path, schedule_path: Path):
        raw = load_raw_dataset(data_dir)
        self._payroll = _prepare_payroll(raw)
        if self._payroll.empty:
            raise ValueError("No payroll transactions available in dataset.")

        self.reference_date = self._payroll["posted_at"].max()
        self._schedules = self._load_schedule_table(schedule_path)
        self._schedule_lookup = self._schedules.set_index("user_id")
        self._frequency_residuals = self._compute_frequency_residual_stats()

    def predict_user(self, user_id: str, reference_date: TimestampLike = None) -> PredictionResult:
        ref_ts = self._resolve_reference(reference_date)
        history = self._user_history(user_id, ref_ts)
        if history.empty:
            raise ValueError(f"No payroll history available for user {user_id!r} before {ref_ts.date()}.")

        schedule = self._schedule_lookup.loc[user_id] if user_id in self._schedule_lookup.index else None
        frequency, anchors = self._resolve_frequency(schedule, history)

        baseline = self._baseline_prediction(history, frequency, anchors, ref_ts, schedule)
        user_residuals = self._user_residuals(history, frequency, anchors, schedule)
        bias_days = self._residual_bias_days(user_residuals, frequency)
        corrected = baseline + pd.Timedelta(days=bias_days) if bias_days else baseline

        low, high = self._interval_bounds(corrected, user_residuals, frequency, schedule)
        return PredictionResult(date=corrected, low=low, high=high)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _load_schedule_table(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Schedule artifact missing: {path}")
        df = pd.read_parquet(path)
        expected_cols = {
            "user_id",
            "history_len",
            "monthly_offset_median",
            "monthly_offset_q10",
            "monthly_offset_q90",
            "last_payroll",
            "declared_frequency",
            "inferred_frequency",
            "median_interval",
            "interval_std",
            "weekday_mode",
            "anchors",
            "reference_date",
        }
        missing = expected_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Schedule table missing columns: {', '.join(sorted(missing))}")
        return df

    def _resolve_reference(self, reference_date: TimestampLike) -> pd.Timestamp:
        if reference_date is None:
            ref = self.reference_date
        elif isinstance(reference_date, pd.Timestamp):
            ref = reference_date
        elif isinstance(reference_date, datetime):
            ref = pd.Timestamp(reference_date)
        elif isinstance(reference_date, str):
            ref = pd.to_datetime(reference_date)
        else:
            raise TypeError("reference_date must be str, datetime, pd.Timestamp, or None.")

        if ref.tzinfo is not None:
            ref = ref.tz_convert("UTC").tz_localize(None)
        else:
            ref = ref.replace(tzinfo=None)
        return ref

    def _user_history(self, user_id: str, reference_date: pd.Timestamp) -> pd.Series:
        history = self._payroll[self._payroll["user_id"] == user_id]
        history = history[history["posted_at"] <= reference_date]
        return history["posted_at"].sort_values().reset_index(drop=True)

    def _resolve_frequency(
        self, schedule_row: Optional[pd.Series], history: pd.Series
    ) -> Tuple[str, List[int]]:
        frequency = "unknown"
        anchors: List[int] = []

        if schedule_row is not None:
            frequency = str(schedule_row["inferred_frequency"])
            anchors = _parse_anchors(str(schedule_row.get("anchors", "")))

        if frequency not in {"weekly", "biweekly", "semimonthly", "monthly"}:
            frequency = self._infer_frequency_from_history(history, anchors)

        if frequency == "semimonthly":
            anchors = self._resolve_semimonthly_anchors(history, anchors)
        elif frequency == "monthly":
            anchors = [self._resolve_monthly_anchor(history, anchors)]
        else:
            anchors = sorted(set(int(a) for a in anchors))

        return frequency, anchors

    @staticmethod
    def _infer_frequency_from_history(history: pd.Series, anchors: Sequence[int]) -> str:
        if len(history) < 2:
            return "unknown"
        deltas = history.diff().dropna().dt.days
        median_interval = float(np.median(deltas))
        if median_interval <= 8:
            return "weekly"
        if median_interval <= 16:
            return "biweekly"
        if median_interval <= 23 and len(anchors) >= 2:
            return "semimonthly"
        return "monthly"

    @staticmethod
    def _default_semimonthly_anchors(history: pd.Series) -> List[int]:
        if history.empty:
            return [15, 30]
        counts = history.dt.day.value_counts().sort_values(ascending=False)
        anchors = counts.index.tolist()[:2]
        anchors = sorted(set(int(day) for day in anchors))
        if len(anchors) == 1:
            anchors.append(min(anchors[0] + 15, 30))
        return anchors[:2]

    def _resolve_semimonthly_anchors(self, history: pd.Series, anchors: Sequence[int]) -> List[int]:
        cleaned = [int(a) for a in anchors if 1 <= int(a) <= 31]
        if len(cleaned) >= 2:
            cleaned = sorted(set(cleaned))[:2]
        if len(cleaned) < 2:
            cleaned = self._default_semimonthly_anchors(history)
        return cleaned

    def _resolve_monthly_anchor(self, history: pd.Series, anchors: Sequence[int]) -> int:
        if history.empty:
            return 30
        days = history.dt.day
        if len(days) >= 3:
            mode = days.mode()
            if not mode.empty:
                return int(mode.iloc[0])
        if anchors:
            anchor_candidates = [int(a) for a in anchors if 1 <= int(a) <= 31]
            if anchor_candidates:
                return int(np.median(anchor_candidates))
        return int(round(float(days.mean())))

    def _baseline_prediction(
        self,
        history: pd.Series,
        frequency: str,
        anchors: Sequence[int],
        reference_date: pd.Timestamp,
        schedule_row: Optional[pd.Series],
    ) -> pd.Timestamp:
        last_date = history.iloc[-1]
        if frequency == "weekly":
            candidate = last_date + pd.Timedelta(days=7)
        elif frequency == "biweekly":
            candidate = last_date + pd.Timedelta(days=14)
        elif frequency == "semimonthly":
            candidate = self._next_semimonthly_date(last_date, anchors)
        elif frequency == "monthly":
            candidate = self._next_monthly_date(reference_date, schedule_row)
        else:
            interval = self._fallback_interval(history)
            candidate = last_date + pd.Timedelta(days=interval)
        return _adjust_weekend(candidate)

    def _next_semimonthly_date(self, last_date: pd.Timestamp, anchors: Sequence[int]) -> pd.Timestamp:
        anchors = sorted(set(int(a) for a in anchors)) or [15, _month_last_day(last_date)]
        anchors = [min(max(a, 1), 31) for a in anchors]
        anchors = sorted(anchors)
        next_anchor = next((a for a in anchors if a > last_date.day), None)
        if next_anchor is not None:
            candidate = _set_day(last_date, next_anchor)
        else:
            next_month = last_date + pd.DateOffset(months=1)
            candidate = _set_day(next_month, anchors[0])
        return candidate

    @staticmethod
    def _last_business_day(period: pd.Period) -> pd.Timestamp:
        anchor = period.to_timestamp(how="end")
        while anchor.weekday() >= 5:
            anchor -= pd.Timedelta(days=1)
        return anchor

    def _next_monthly_date(self, reference_date: pd.Timestamp, schedule_row: Optional[pd.Series]) -> pd.Timestamp:
        ref = pd.Timestamp(reference_date)
        if ref.tzinfo is not None:
            ref = ref.tz_convert("UTC").tz_localize(None)
        else:
            ref = ref.replace(tzinfo=None)
        anchor_period = ref.to_period("M")
        if ref.day <= 7:
            anchor_period = (ref - pd.DateOffset(months=1)).to_period("M")
        period = anchor_period + 1
        anchor = self._last_business_day(period)
        offset = 0.0
        if schedule_row is not None:
            value = schedule_row.get("monthly_offset_median")
            if value is not None and not pd.isna(value):
                offset = float(value)
        candidate = anchor + pd.Timedelta(days=offset)
        return candidate

    @staticmethod
    def _fallback_interval(history: pd.Series) -> int:
        if len(history) < 2:
            return 14
        deltas = history.diff().dropna().dt.days
        return int(round(float(np.median(deltas))))

    def _user_residuals(
        self,
        history: pd.Series,
        frequency: str,
        anchors: Sequence[int],
        schedule_row: Optional[pd.Series],
    ) -> List[float]:
        if len(history) < 2:
            return []
        residuals: List[float] = []
        for idx in range(1, len(history)):
            prior = history.iloc[:idx]
            actual = history.iloc[idx]
            reference_date = prior.iloc[-1]
            baseline = self._baseline_prediction(prior, frequency, anchors, reference_date, schedule_row)
            residuals.append((actual - baseline).days)
        return residuals

    def _residual_bias_days(self, residuals: List[float], frequency: str) -> float:
        if frequency == "monthly":
            return 0.0
        if residuals:
            window = residuals[-6:] if len(residuals) >= 6 else residuals
            bias = float(np.median(window))
            return float(np.clip(bias, -4.0, 4.0))
        stats = self._frequency_residuals.get(frequency)
        if stats:
            median = stats.get("median")
            if median is not None:
                return float(np.clip(median, -4.0, 4.0))
        return 0.0

    def _interval_bounds(
        self,
        center: pd.Timestamp,
        residuals: List[float],
        frequency: str,
        schedule_row: Optional[pd.Series],
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        if frequency == "monthly" and schedule_row is not None:
            median = schedule_row.get("monthly_offset_median")
            q10 = schedule_row.get("monthly_offset_q10")
            q90 = schedule_row.get("monthly_offset_q90")
            low = high = None
            if median is not None and q10 is not None and not pd.isna(median) and not pd.isna(q10):
                delta = float(q10) - float(median)
                low = center + pd.Timedelta(days=delta)
            if median is not None and q90 is not None and not pd.isna(median) and not pd.isna(q90):
                delta = float(q90) - float(median)
                high = center + pd.Timedelta(days=delta)
        else:
            low_offset, high_offset = self._residual_quantiles(residuals, frequency)
            low = center + pd.Timedelta(days=low_offset) if low_offset is not None else None
            high = center + pd.Timedelta(days=high_offset) if high_offset is not None else None
        if low is not None:
            low = _adjust_weekend(low)
        if high is not None:
            high = _adjust_weekend(high)
        if low is not None and high is not None and high < low:
            low, high = high, low
        return low, high

    def _residual_quantiles(
        self, residuals: List[float], frequency: str
    ) -> Tuple[Optional[float], Optional[float]]:
        values = residuals[-10:] if len(residuals) >= 3 else []
        if values:
            low = float(np.quantile(values, 0.10))
            high = float(np.quantile(values, 0.90))
            return low, high
        stats = self._frequency_residuals.get(frequency)
        if stats:
            return stats["q10"], stats["q90"]
        return None, None

    def _compute_frequency_residual_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        residuals: Dict[str, List[float]] = {freq: [] for freq in ["weekly", "biweekly", "semimonthly", "monthly"]}
        for user_id, group in self._payroll.groupby("user_id"):
            schedule = self._schedule_lookup.loc[user_id] if user_id in self._schedule_lookup.index else None
            history = group["posted_at"].sort_values().reset_index(drop=True)
            frequency, anchors = self._resolve_frequency(schedule, history)
            if frequency not in residuals:
                continue
            res = self._user_residuals(history, frequency, anchors, schedule)
            residuals[frequency].extend(res)

        for freq, values in residuals.items():
            if values:
                stats[freq] = {
                    "median": float(np.median(values)),
                    "q10": float(np.quantile(values, 0.10)),
                    "q90": float(np.quantile(values, 0.90)),
                }
            else:
                stats[freq] = {"median": None, "q10": None, "q90": None}
        return stats


def load_predictor(data_dir: Path, schedule_path: Path) -> PaydatePredictor:
    return PaydatePredictor(data_dir=data_dir, schedule_path=schedule_path)
