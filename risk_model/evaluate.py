"""Evaluation entrypoint for the repayment risk model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from risk_model.datasets import load_raw_dataset
from risk_model.features import build_feature_matrix
from risk_model.splits import assign_time_split

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the repayment risk model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the generated dataset directory.")
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory containing trained model artifacts.")
    parser.add_argument("--out", type=Path, default=Path("./artifacts/risk"), help="Directory for evaluation artifacts.")
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins to use for calibration diagnostics.",
    )
    parser.add_argument(
        "--approval-targets",
        type=float,
        nargs="+",
        default=(0.4, 0.6, 0.8),
        help="Approval rate targets for business simulation (expressed as fractions).",
    )
    parser.add_argument(
        "--fee-pct",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.03),
        help="Fee percentages used in business simulation.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    out_dir = args.out.expanduser().resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset from %s", args.data)
    raw = load_raw_dataset(args.data)
    features, labels, feature_artifacts = build_feature_matrix(raw)

    split_series = assign_time_split(raw.advances)
    split_by_id = pd.Series(split_series.values, index=raw.advances["advance_id"], name="split")
    dataset = features.join(split_by_id, how="inner")
    labels = labels.loc[dataset.index]

    LOGGER.info("Loading trained pipeline from %s", args.artifacts)
    pipeline_path = (args.artifacts / "model.joblib").resolve()
    pipeline = joblib.load(pipeline_path)

    training_summary_path = args.artifacts / "training_summary.json"
    training_summary = None
    if training_summary_path.exists():
        with training_summary_path.open("r", encoding="utf-8") as f:
            training_summary = json.load(f)

    metrics_by_split: Dict[str, Dict[str, float]] = {}
    proba_by_split: Dict[str, pd.Series] = {}

    for split in ("train", "val", "test"):
        mask = split_by_id == split
        if mask.sum() == 0:
            LOGGER.warning("Split %s is empty; skipping.", split)
            continue

        X_split = dataset.loc[mask].drop(columns=["split"])
        y_split = labels.loc[X_split.index]
        proba_split = pipeline.predict_proba(X_split)[:, 1]
        proba_series = pd.Series(proba_split, index=X_split.index, name="p_bad")

        metrics_by_split[split] = _compute_metrics(y_split, proba_split, args.calibration_bins)
        proba_by_split[split] = proba_series
        LOGGER.info(
            "%s split: ROC-AUC=%.3f PR-AUC=%.3f Brier=%.4f",
            split.capitalize(),
            metrics_by_split[split]["roc_auc"],
            metrics_by_split[split]["pr_auc"],
            metrics_by_split[split]["brier"],
        )

    if "test" not in metrics_by_split:
        raise ValueError("Test split evaluation failed; ensure the dataset contains a test partition.")

    X_test = dataset.loc[split_by_id == "test"].drop(columns=["split"])
    y_test = labels.loc[X_test.index]
    proba_test = proba_by_split["test"]

    calibration_table = _calibration_table(y_test, proba_test, args.calibration_bins)
    calibration_csv = out_dir / "calibration_test.csv"
    calibration_table.to_csv(calibration_csv, index=False)
    _plot_calibration(calibration_table, plots_dir / "calibration_test.png")

    slice_pay_freq = _group_metrics(
        X_test,
        y_test,
        proba_test,
        group_col="pay_frequency",
    )
    slice_pay_freq_path = out_dir / "slice_pay_frequency.csv"
    slice_pay_freq.to_csv(slice_pay_freq_path, index=False)

    slice_amount_deciles = _group_metrics(
        X_test.assign(amount_decile=_amount_deciles(X_test["advance_amount_cents"])),
        y_test,
        proba_test,
        group_col="amount_decile",
    )
    slice_amount_path = out_dir / "slice_amount_deciles.csv"
    slice_amount_deciles.to_csv(slice_amount_path, index=False)

    business_sim = _business_simulation(
        y_test,
        proba_test,
        X_test["advance_amount_cents"],
        approval_targets=args.approval_targets,
        fee_pct=args.fee_pct,
    )
    business_sim_path = out_dir / "business_simulation.csv"
    business_sim.to_csv(business_sim_path, index=False)

    cohort_stats = _cohort_stats(raw.advances, labels)
    cohort_stats_path = out_dir / "cohort_stats.csv"
    cohort_stats.to_csv(cohort_stats_path, index=False)

    evaluation_summary = {
        "metrics": metrics_by_split,
        "calibration_bins": args.calibration_bins,
        "approval_targets": args.approval_targets,
        "fee_pct": args.fee_pct,
        "artifacts": {
            "calibration": str(calibration_csv),
            "calibration_plot": str(plots_dir / "calibration_test.png"),
            "slice_pay_frequency": str(slice_pay_freq_path),
            "slice_amount_deciles": str(slice_amount_path),
            "business_simulation": str(business_sim_path),
            "cohort_stats": str(cohort_stats_path),
        },
        "training_summary": training_summary,
        "feature_metadata": {
            "numeric_columns": feature_artifacts.numeric_columns,
            "categorical_columns": feature_artifacts.categorical_columns,
            **feature_artifacts.metadata,
        },
    }

    summary_path = out_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    LOGGER.info("Evaluation complete. Summary written to %s", summary_path)


def _compute_metrics(
    y_true: Iterable[int],
    proba: Iterable[float],
    calibration_bins: int,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "ece": float(_expected_calibration_error(y_true, proba, calibration_bins)),
    }


def _expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bins) - 1
    total = len(proba)
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        avg_conf = proba[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += np.abs(avg_conf - avg_acc) * mask.sum()
    return ece / total if total else 0.0


def _calibration_table(
    y_true: pd.Series,
    proba: pd.Series,
    n_bins: int,
) -> pd.DataFrame:
    bin_codes, bin_edges = pd.qcut(
        proba,
        q=n_bins,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    effective_bins = len(bin_edges) - 1
    prob_true, prob_pred = calibration_curve(
        y_true,
        proba,
        n_bins=effective_bins,
        strategy="quantile",
    )
    bin_codes = bin_codes.astype(int)
    counts = np.bincount(bin_codes, minlength=effective_bins)
    return pd.DataFrame(
        {
            "bin_lower": bin_edges[:-1],
            "bin_upper": bin_edges[1:],
            "count": counts,
            "avg_predicted": prob_pred,
            "avg_observed": prob_true,
        }
    )


def _plot_calibration(calibration_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    plt.plot(calibration_df["avg_predicted"], calibration_df["avg_observed"], marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve (test split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _group_metrics(
    X: pd.DataFrame,
    y_true: pd.Series,
    proba: pd.Series,
    group_col: str,
) -> pd.DataFrame:
    groups = X[group_col].astype("object").fillna("Missing")
    proba = proba.astype(float)
    records = []

    for group_value, indices in groups.groupby(groups).groups.items():
        mask_index = list(indices)
        y_group = y_true.loc[mask_index]
        p_group = proba.loc[mask_index]
        if len(y_group) == 0:
            continue
        metrics = _compute_metrics(y_group, p_group, calibration_bins=5)
        metrics.update(
            {
                group_col: group_value,
                "count": int(len(y_group)),
                "positive_rate_pct": float(y_group.mean() * 100),
            }
        )
        records.append(metrics)

    return pd.DataFrame(records)


def _amount_deciles(amounts: pd.Series, q: int = 10) -> pd.Series:
    try:
        deciles = pd.qcut(amounts, q=q, labels=[f"decile_{i+1}" for i in range(q)], duplicates="drop")
    except ValueError:
        deciles = pd.Series(["decile_1"] * len(amounts), index=amounts.index)
    return deciles


def _business_simulation(
    y_true: pd.Series,
    proba: pd.Series,
    amounts: pd.Series,
    approval_targets: Sequence[float],
    fee_pct: Sequence[float],
) -> pd.DataFrame:
    results = []
    n = len(proba)
    proba_sorted = np.sort(proba)
    for target in approval_targets:
        threshold = np.quantile(proba_sorted, target)
        approved_mask = proba <= threshold
        approval_pct = float(approved_mask.mean() * 100)
        if approved_mask.sum() == 0:
            continue
        bad_rate_pct = float(y_true[approved_mask].mean() * 100)
        avg_amount = float(amounts[approved_mask].mean())
        entry = {
            "target_approval_pct": target * 100,
            "actual_approval_pct": approval_pct,
            "threshold": float(threshold),
            "bad_rate_pct": bad_rate_pct,
            "avg_amount_cents": avg_amount,
        }
        for fee in fee_pct:
            entry[f"expected_loss_pct_fee_{int(fee*100)}"] = bad_rate_pct - (fee * 100)
        results.append(entry)
    return pd.DataFrame(results)


def _cohort_stats(advances: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    advances = advances.copy()
    advances = advances.assign(bad_outcome=labels.reindex(advances["advance_id"]).values)
    advances["month"] = advances["requested_at"].dt.tz_convert("UTC").dt.to_period("M").astype(str)
    cohort = (
        advances.groupby("month")
        .agg(
            advances=("advance_id", "count"),
            bad_rate_pct=("bad_outcome", lambda x: float(np.nanmean(x) * 100)),
            avg_amount_cents=("amount_cents", "mean"),
        )
        .reset_index()
    )
    return cohort


if __name__ == "__main__":
    main()
