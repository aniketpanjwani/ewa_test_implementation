"""Evaluation CLI for pay-date prediction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from risk_model.datasets import load_raw_dataset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __package__ is None or __package__ == "":
    sys.path.append(str(_project_root()))
    from paydate_model.predictor import PaydatePredictor, PredictionResult  # type: ignore
else:  # pragma: no cover
    from .predictor import PaydatePredictor, PredictionResult


@dataclass
class EvaluationConfig:
    data_dir: Path
    artifacts_dir: Path
    out_dir: Path


def _prepare_payroll(raw) -> pd.DataFrame:
    payroll = raw.transactions[raw.transactions["category"] == "payroll"].copy()
    if payroll.empty:
        raise ValueError("No payroll transactions found for evaluation.")
    payroll["posted_at"] = payroll["posted_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    payroll.sort_values(["user_id", "posted_at"], inplace=True, ignore_index=True)
    payroll["month"] = payroll["posted_at"].dt.to_period("M")
    return payroll[["user_id", "posted_at", "month"]]


def _split_months(months: Sequence[pd.Period]) -> Tuple[List[pd.Period], List[pd.Period], List[pd.Period]]:
    ordered = sorted(months)
    if len(ordered) < 18:
        raise ValueError("Expected at least 18 months of data to match challenge spec.")
    train = list(ordered[:12])
    val = list(ordered[12:15])
    test = list(ordered[15:18])
    return train, val, test


def _predict_rows(
    predictor: PaydatePredictor, payroll: pd.DataFrame, test_months: Iterable[pd.Period], user_freq: Dict[str, str]
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    test_set = set(test_months)

    for user_id, frame in payroll.groupby("user_id", sort=False):
        frame = frame.sort_values("posted_at").reset_index(drop=True)
        declared = user_freq.get(user_id, "unknown")

        for idx in range(1, len(frame)):
            actual_date = frame.at[idx, "posted_at"]
            if frame.at[idx, "month"] not in test_set:
                continue
            reference_date = frame.at[idx - 1, "posted_at"]
            # Predict using history available up to reference date.
            result: PredictionResult = predictor.predict_user(user_id, reference_date=reference_date)
            error_days = (result.date - actual_date).days
            abs_error = abs(error_days)
            low, high = result.low, result.high
            covered = None
            if low is not None and high is not None:
                covered = int(low <= actual_date <= high)

            records.append(
                {
                    "user_id": user_id,
                    "declared_frequency": declared,
                    "reference_date": reference_date.date().isoformat(),
                    "actual_date": actual_date.date().isoformat(),
                    "predicted_date": result.date.date().isoformat(),
                    "low": low.date().isoformat() if low is not None else None,
                    "high": high.date().isoformat() if high is not None else None,
                    "error_days": error_days,
                    "abs_error": abs_error,
                    "hit_1": int(abs_error <= 1),
                    "hit_2": int(abs_error <= 2),
                    "covered": covered,
                }
            )
    if not records:
        raise ValueError("No evaluation records generated for the selected test window.")
    return pd.DataFrame.from_records(records)


def _aggregate_metrics(df: pd.DataFrame) -> Dict[str, object]:
    coverage_mask = df["covered"].notna()
    coverage = None
    if coverage_mask.any():
        coverage = float(df.loc[coverage_mask, "covered"].mean())

    overall = {
        "count": int(df.shape[0]),
        "mae": float(df["abs_error"].mean()),
        "hit_rate_1": float(df["hit_1"].mean()),
        "hit_rate_2": float(df["hit_2"].mean()),
        "interval_coverage": coverage,
    }
    return overall


def _metrics_by_frequency(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("declared_frequency")
        .agg(
            count=("user_id", "size"),
            mae=("abs_error", "mean"),
            hit_rate_1=("hit_1", "mean"),
            hit_rate_2=("hit_2", "mean"),
            interval_coverage=("covered", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
        )
        .reset_index()
    )
    numeric_cols = ["mae", "hit_rate_1", "hit_rate_2", "interval_coverage"]
    for col in numeric_cols:
        grouped[col] = grouped[col].astype(float)
    return grouped


def _write_outputs(out_dir: Path, predictions: pd.DataFrame, overall: Dict[str, object], freq_metrics: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    freq_metrics.to_csv(out_dir / "metrics_by_frequency.csv", index=False)
    with (out_dir / "metrics_overall.json").open("w", encoding="utf-8") as fh:
        json.dump(overall, fh, indent=2)


def _print_summary(overall: Dict[str, object], freq_metrics: pd.DataFrame) -> None:
    print("=== Pay-Date Prediction Evaluation ===")
    print(json.dumps(overall, indent=2))
    print("\nMetrics by pay frequency:")
    print(freq_metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def run_evaluation(config: EvaluationConfig) -> None:
    raw = load_raw_dataset(config.data_dir)
    payroll = _prepare_payroll(raw)
    _, _, test_months = _split_months(payroll["month"].unique())

    schedule_path = config.artifacts_dir / "schedules.parquet"
    predictor = PaydatePredictor(data_dir=config.data_dir, schedule_path=schedule_path)

    user_freq = {row.user_id: str(row.pay_frequency).lower() for row in raw.users.itertuples()}
    predictions = _predict_rows(predictor, payroll, test_months, user_freq)
    overall = _aggregate_metrics(predictions)
    freq_metrics = _metrics_by_frequency(predictions)

    _write_outputs(config.out_dir, predictions, overall, freq_metrics)
    _print_summary(overall, freq_metrics)


def parse_args(argv: Optional[Sequence[str]] = None) -> EvaluationConfig:
    default_root = _project_root()
    parser = argparse.ArgumentParser(description="Evaluate next-payday predictions on held-out months.")
    parser.add_argument("--data", type=Path, default=default_root / "data", help="Path to dataset directory.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=default_root / "artifacts" / "paydate",
        help="Path to paydate artifact directory containing schedules.parquet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_root / "artifacts" / "paydate" / "eval",
        help="Output directory for evaluation metrics.",
    )
    args = parser.parse_args(argv)
    return EvaluationConfig(data_dir=args.data.resolve(), artifacts_dir=args.artifacts.resolve(), out_dir=args.out.resolve())


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    run_evaluation(config)


if __name__ == "__main__":
    main()
