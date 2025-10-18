from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from risk_model.datasets import load_raw_dataset
from risk_model.features import build_feature_matrix
from risk_model.splits import assign_time_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stacked ensemble using risk model outputs.")
    parser.add_argument("--base-artifacts", type=Path, nargs="+", required=True, help="Base model artifact directories.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset(args.data)
    features, labels, _ = build_feature_matrix(raw)
    splits = assign_time_split(raw.advances)
    split_by_id = pd.Series(splits.values, index=features.index, name="split")

    meta_features = {}
    for artifact_dir in args.base_artifacts:
        artifact_dir = artifact_dir.resolve()
        model = joblib.load(artifact_dir / "model.joblib")
        proba = pd.Series(model.predict_proba(features)[:, 1], index=features.index, name=artifact_dir.name)
        meta_features[artifact_dir.name] = proba

    stack_df = pd.DataFrame(meta_features).join(split_by_id, how="inner").join(labels.rename("label"))
    X_train = stack_df[stack_df["split"] == "train"].drop(columns=["split", "label"])
    y_train = stack_df.loc[X_train.index, "label"]
    X_val = stack_df[stack_df["split"] == "val"].drop(columns=["split", "label"])
    y_val = stack_df.loc[X_val.index, "label"]
    X_test = stack_df[stack_df["split"] == "test"].drop(columns=["split", "label"])
    y_test = stack_df.loc[X_test.index, "label"]

    base_metrics = {col: _metrics(y_val, X_val[col]) for col in X_val.columns}

    stacker = LogisticRegression(max_iter=1000, random_state=args.seed)
    stacker.fit(X_train, y_train)

    stack_metrics = {
        "val": _metrics(y_val, stacker.predict_proba(X_val)[:, 1]),
        "test": _metrics(y_test, stacker.predict_proba(X_test)[:, 1]),
    }

    joblib.dump(stacker, args.out / "stacker.joblib")
    with (args.out / "stack_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"base_metrics": base_metrics, "stack_metrics": stack_metrics}, f, indent=2)


def _metrics(y_true: pd.Series, proba: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


if __name__ == "__main__":
    main()
