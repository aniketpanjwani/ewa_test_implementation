"""Payroll cadence inference utilities for pay-date prediction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from risk_model.datasets import RawDataset, load_raw_dataset


PAY_FREQUENCIES = ("weekly", "biweekly", "semimonthly", "monthly")


@dataclass
class ScheduleRecord:
    """Container describing a user's inferred payroll cadence."""

    user_id: str
    history_len: int
    monthly_offset_median: Optional[float]
    monthly_offset_q10: Optional[float]
    monthly_offset_q90: Optional[float]
    last_payroll: Optional[pd.Timestamp]
    declared_frequency: Optional[str]
    inferred_frequency: str
    median_interval: Optional[float]
    interval_std: Optional[float]
    weekday_mode: Optional[str]
    anchors: List[int]
    reference_date: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "history_len": self.history_len,
            "monthly_offset_median": self.monthly_offset_median,
            "monthly_offset_q10": self.monthly_offset_q10,
            "monthly_offset_q90": self.monthly_offset_q90,
            "last_payroll": (
                self.last_payroll.strftime("%Y-%m-%d") if self.last_payroll is not None else None
            ),
            "declared_frequency": self.declared_frequency,
            "inferred_frequency": self.inferred_frequency,
            "median_interval": self.median_interval,
            "interval_std": self.interval_std,
            "weekday_mode": self.weekday_mode,
            "anchors": ",".join(str(a) for a in self.anchors) if self.anchors else "",
            "reference_date": self.reference_date.strftime("%Y-%m-%d"),
        }


def _prepare_payroll(raw: RawDataset) -> pd.DataFrame:
    """Return sorted payroll transactions with UTC timestamps."""

    payroll = raw.transactions[raw.transactions["category"] == "payroll"].copy()
    if payroll.empty:
        return pd.DataFrame(columns=["user_id", "posted_at", "posted_naive"])
    payroll["posted_at"] = payroll["posted_at"].dt.tz_convert("UTC")
    payroll.sort_values(["user_id", "posted_at"], inplace=True, ignore_index=True)
    payroll["posted_naive"] = payroll["posted_at"].dt.tz_localize(None)
    return payroll[["user_id", "posted_at", "posted_naive"]]


def _weekday_mode(dates: pd.Series) -> Optional[str]:
    if dates.empty:
        return None
    mode = dates.dt.day_name().mode()
    return mode.iat[0] if not mode.empty else None


def _interval_stats(dates: pd.Series) -> tuple[Optional[float], Optional[float]]:
    if len(dates) < 2:
        return None, None
    deltas = dates.diff().dropna().dt.days.to_numpy()
    return float(np.median(deltas)), float(np.std(deltas))


def _anchor_days(dates: pd.Series) -> List[int]:
    if dates.empty:
        return []
    counts = Counter(int(day) for day in dates.dt.day)
    anchors = [day for day, _ in counts.most_common(2)]
    anchors = sorted(set(anchors))
    return anchors


def _infer_frequency(median_interval: Optional[float], anchors: Iterable[int]) -> str:
    if median_interval is None or np.isnan(median_interval):
        return "unknown"
    anchor_count = len(list(anchors))
    if median_interval <= 8:
        return "weekly"
    if median_interval <= 16:
        return "biweekly"
    if median_interval <= 23 and anchor_count >= 2:
        return "semimonthly"
    return "monthly"


def _last_business_day(period: pd.Period) -> pd.Timestamp:
    anchor = period.to_timestamp(how="end")
    while anchor.weekday() >= 5:
        anchor -= pd.Timedelta(days=1)
    return anchor


def _monthly_offset_stats(dates: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if dates.empty or len(dates) < 2:
        return None, None, None
    offsets: List[int] = []
    for value in dates.dropna():
        ts = pd.Timestamp(value)
        period = ts.to_period("M")
        if ts.day <= 7:
            period = (ts - pd.DateOffset(months=1)).to_period("M")
        anchor = _last_business_day(period)
        offset = int((ts.normalize() - anchor.normalize()).days)
        offsets.append(offset)
    if not offsets:
        return None, None, None
    arr = np.clip(np.asarray(offsets, dtype=float), -7, 7)
    median = float(np.median(arr))
    q10 = float(np.quantile(arr, 0.10))
    q90 = float(np.quantile(arr, 0.90))
    return median, q10, q90


def _collect_schedules(raw: RawDataset) -> list[ScheduleRecord]:
    payroll = _prepare_payroll(raw)
    reference_date = payroll["posted_naive"].max() if not payroll.empty else pd.Timestamp.utcnow()
    users = raw.users[["user_id", "pay_frequency"]].set_index("user_id")

    records: list[ScheduleRecord] = []
    for user_id, frame in payroll.groupby("user_id"):
        dates = frame["posted_naive"]
        history_len = len(dates)
        last_payroll = dates.iloc[-1] if history_len else None
        median_interval, interval_std = _interval_stats(dates)
        anchors = _anchor_days(dates)
        frequency = _infer_frequency(median_interval, anchors)
        weekday = _weekday_mode(dates)
        declared = None
        if user_id in users.index:
            declared = str(users.at[user_id, "pay_frequency"]).lower()
        offset_median = offset_q10 = offset_q90 = None
        if frequency == "monthly" or declared == "monthly":
            offset_median, offset_q10, offset_q90 = _monthly_offset_stats(frame["posted_naive"])
        record = ScheduleRecord(
            user_id=user_id,
            history_len=history_len,
            monthly_offset_median=None if offset_median is None else round(offset_median, 3),
            monthly_offset_q10=None if offset_q10 is None else round(offset_q10, 3),
            monthly_offset_q90=None if offset_q90 is None else round(offset_q90, 3),
            last_payroll=last_payroll,
            declared_frequency=declared,
            inferred_frequency=frequency,
            median_interval=None if median_interval is None else round(median_interval, 3),
            interval_std=None if interval_std is None else round(interval_std, 3),
            weekday_mode=weekday,
            anchors=anchors,
            reference_date=reference_date,
        )
        records.append(record)

    return records


def build_schedule_table(data_dir: Path) -> pd.DataFrame:
    """Generate the per-user cadence table."""

    raw = load_raw_dataset(data_dir)
    records = _collect_schedules(raw)
    if not records:
        return pd.DataFrame(
            columns=[
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
            ]
        )
    return pd.DataFrame(record.to_dict() for record in records)


def _write_metadata(out_dir: Path, table: pd.DataFrame) -> None:
    meta = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": int(table.shape[0]),
        "columns": table.columns.tolist(),
    }
    with (out_dir / "schedules_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def main(argv: Optional[list[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="Infer user payroll cadences and persist schedules.")
    parser.add_argument("--data", type=Path, default=Path("./data"), help="Path to synthetic dataset directory.")
    parser.add_argument("--out", type=Path, default=Path("./artifacts/paydate"), help="Output directory for artifacts.")
    args = parser.parse_args(argv)

    table = build_schedule_table(args.data)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "schedules.parquet"
    table.to_parquet(parquet_path, index=False)
    _write_metadata(out_dir, table)

    print(f"Wrote {len(table)} user schedules to {parquet_path}")
    return parquet_path


if __name__ == "__main__":
    main()
