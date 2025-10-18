"""Hyperparameter search for the LightGBM risk model focused on ROC-AUC."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from risk_model.datasets import load_raw_dataset
from risk_model.features import FeatureArtifacts, build_feature_matrix
from risk_model.splits import assign_time_split
from risk_model.train import _build_preprocessor

LOGGER = logging.getLogger(__name__)


@dataclass
class TrialResult:
    params: Dict[str, object]
    mean_roc_auc: float
    mean_pr_auc: float
    fold_scores: List[Tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-aware LightGBM hyperparameter search optimising ROC-AUC.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the generated dataset directory.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random search trials to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./artifacts/risk/lightgbm_tuning.json"),
        help="Path to write tuning results (JSON).",
    )
    return parser.parse_args()


def _prepare_training_frame(data_dir: Path, seed: int) -> Tuple[pd.DataFrame, pd.Series, FeatureArtifacts]:
    raw = load_raw_dataset(data_dir)
    features, labels, artifacts = build_feature_matrix(raw)

    split_series = assign_time_split(raw.advances)
    split_by_id = pd.Series(split_series.values, index=raw.advances["advance_id"], name="split")
    dataset = features.join(split_by_id, how="inner")
    labels = labels.loc[dataset.index]

    # Restrict to train split; keep chronological order via requested_at
    train_mask = dataset["split"] == "train"
    train_df = dataset.loc[train_mask].copy()
    train_df = train_df.sort_values("requested_at")
    train_labels = labels.loc[train_df.index]

    train_df = train_df.drop(columns=["split"])

    return train_df, train_labels, artifacts


def _sample_parameters(rng: random.Random) -> Dict[str, object]:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": rng.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.05]),
        "n_estimators": rng.choice([400, 600, 800, 1000]),
        "num_leaves": rng.randint(31, 512),
        "max_depth": rng.choice([-1, 4, 5, 6, 7, 8, 9, 10]),
        "min_child_samples": rng.randint(10, 160),
        "subsample": round(rng.uniform(0.5, 1.0), 2),
        "subsample_freq": rng.choice([0, 1, 2, 3, 4, 5]),
        "colsample_bytree": round(rng.uniform(0.5, 1.0), 2),
        "reg_lambda": 10 ** rng.uniform(-4, 2),
        "reg_alpha": 10 ** rng.uniform(-4, 1),
        "min_split_gain": round(rng.uniform(0.0, 0.5), 3),
        "max_bin": rng.choice([127, 255, 383, 511, 767]),
        "n_jobs": -1,
        "verbose": -1,
        "random_state": rng.randint(0, 10_000),
    }


def _evaluate_params(
    params: Dict[str, object],
    features: pd.DataFrame,
    labels: pd.Series,
    artifacts: FeatureArtifacts,
    seed: int,
    folds: int = 4,
) -> TrialResult:
    splitter = TimeSeriesSplit(n_splits=folds)
    indices = np.arange(len(features))

    fold_scores: List[Tuple[float, float]] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(indices), start=1):
        X_train = features.iloc[train_idx]
        y_train = labels.iloc[train_idx]
        X_val = features.iloc[val_idx]
        y_val = labels.iloc[val_idx]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(artifacts)),
                (
                    "classifier",
                    LGBMClassifier(**params),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, proba)
        pr = average_precision_score(y_val, proba)
        fold_scores.append((roc, pr))
        LOGGER.info(
            "Params %s — fold %d ROC-AUC=%.3f PR-AUC=%.3f",
            params.get("random_state"),
            fold,
            roc,
            pr,
        )

    roc_scores = [roc for roc, _ in fold_scores]
    pr_scores = [pr for _, pr in fold_scores]
    return TrialResult(
        params=params,
        mean_roc_auc=float(np.mean(roc_scores)),
        mean_pr_auc=float(np.mean(pr_scores)),
        fold_scores=fold_scores,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    rng = random.Random(args.seed)
    LOGGER.info("Loading dataset from %s", args.data)
    features, labels, artifacts = _prepare_training_frame(args.data, args.seed)
    LOGGER.info("Training rows for tuning: %d", len(features))

    best_result: Optional[TrialResult] = None
    all_results: List[Dict[str, object]] = []

    for trial in range(1, args.trials + 1):
        params = _sample_parameters(rng)
        LOGGER.info("Trial %d/%d — testing params: %s", trial, args.trials, params)
        result = _evaluate_params(params, features, labels, artifacts, args.seed)
        all_results.append(
            {
                "trial": trial,
                "params": result.params,
                "mean_roc_auc": result.mean_roc_auc,
                "mean_pr_auc": result.mean_pr_auc,
                "fold_scores": [{"roc_auc": roc, "pr_auc": pr} for roc, pr in result.fold_scores],
            }
        )
        if not best_result or result.mean_roc_auc > best_result.mean_roc_auc:
            best_result = result
            LOGGER.info(
                "New best — mean ROC-AUC=%.4f PR-AUC=%.4f (params random_state=%s)",
                result.mean_roc_auc,
                result.mean_pr_auc,
                result.params.get("random_state"),
            )

    if best_result is None:
        raise RuntimeError("No tuning trials completed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "trials": args.trials,
        "best_params": best_result.params,
        "best_mean_roc_auc": best_result.mean_roc_auc,
        "best_mean_pr_auc": best_result.mean_pr_auc,
        "results": all_results,
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info(
        "Tuning complete. Best mean ROC-AUC=%.4f PR-AUC=%.4f. Params saved to %s",
        best_result.mean_roc_auc,
        best_result.mean_pr_auc,
        args.out,
    )


if __name__ == "__main__":
    main()
