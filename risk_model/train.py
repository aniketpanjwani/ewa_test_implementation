"""Training entrypoint for the repayment risk model."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from risk_model.datasets import load_raw_dataset
from risk_model.features import FeatureArtifacts, build_feature_matrix
from risk_model.splits import assign_time_split, compute_split_boundaries, summarize_splits

LOGGER = logging.getLogger(__name__)

MODEL_BUILDERS = {
    "logistic_regression": lambda seed: LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    ),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=None,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=0.01,
        random_state=seed,
    ),
    "hist_gradient_boosting_tuned": lambda seed: HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_depth=6,
        max_iter=550,
        min_samples_leaf=15,
        l2_regularization=0.0,
        validation_fraction=None,
        random_state=seed,
    ),
    "xgboost": lambda seed: XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=400,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        verbosity=0,
        n_jobs=0,
    ),
    "xgboost_tuned": lambda seed: XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        n_estimators=600,
        reg_lambda=0.5,
        reg_alpha=0.1,
        random_state=seed,
        verbosity=0,
        n_jobs=0,
    ),
}

MODEL_REGISTRY = tuple(MODEL_BUILDERS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the repayment risk model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the generated dataset directory.")
    parser.add_argument("--out", type=Path, required=True, help="Directory where model artifacts will be written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_REGISTRY,
        default=None,
        help="Subset of models to train (defaults to all).",
    )
    parser.add_argument(
        "--calibration",
        choices=("isotonic", "sigmoid", "none"),
        default="isotonic",
        help="Probability calibration method applied to the selected model.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading raw dataset from %s", args.data)
    raw = load_raw_dataset(args.data)
    features, labels, feature_artifacts = build_feature_matrix(raw)

    split_series = assign_time_split(raw.advances)
    split_by_id = pd.Series(split_series.values, index=raw.advances["advance_id"], name="split")
    dataset = features.join(split_by_id, how="inner")
    labels = labels.loc[dataset.index]

    LOGGER.info("Dataset size by split:\n%s", summarize_splits(split_by_id))

    X_train, y_train = _subset_split(dataset, labels, "train")
    X_val, y_val = _subset_split(dataset, labels, "val")
    X_test, y_test = _subset_split(dataset, labels, "test")

    if X_val.empty or X_test.empty:
        raise ValueError("Validation and test splits must be non-empty to proceed.")

    LOGGER.info("Fitting preprocessing pipeline")
    preprocessor = _build_preprocessor(feature_artifacts)
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train).astype(np.float32)
    X_val_proc = preprocessor.transform(X_val).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test).astype(np.float32)

    requested_models = args.models or MODEL_REGISTRY
    LOGGER.info("Training candidate models: %s", ", ".join(requested_models))

    model_results: List[Dict[str, object]] = []
    for model_name in requested_models:
        estimator = _instantiate_model(model_name, args.seed)
        fitted = clone(estimator).fit(X_train_proc, y_train)
        val_proba = fitted.predict_proba(X_val_proc)[:, 1]
        metrics = _compute_metrics(y_val, val_proba)
        model_results.append({"model": model_name, "val_metrics": metrics})
        LOGGER.info(
            "Model %s validation ROC-AUC=%.3f PR-AUC=%.3f Brier=%.4f",
            model_name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["brier"],
        )

    best_model_name = max(model_results, key=lambda r: r["val_metrics"]["roc_auc"])["model"]
    LOGGER.info("Selected model: %s", best_model_name)

    final_preprocessor = _build_preprocessor(feature_artifacts)
    final_estimator = _instantiate_model(best_model_name, args.seed)
    calibration_method = args.calibration.lower()

    if calibration_method != "none":
        cv_folds = _determine_calibration_folds(y_train)
        LOGGER.info("Applying %s calibration with %d-fold CV", calibration_method, cv_folds)
        final_estimator = CalibratedClassifierCV(
            estimator=final_estimator,
            method=calibration_method,
            cv=cv_folds,
        )
    else:
        LOGGER.info("Skipping probability calibration")

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", final_preprocessor),
            ("classifier", final_estimator),
        ]
    )
    final_pipeline.fit(X_train, y_train)

    metrics_by_split = {
        "train": _compute_metrics(y_train, final_pipeline.predict_proba(X_train)[:, 1]),
        "val": _compute_metrics(y_val, final_pipeline.predict_proba(X_val)[:, 1]),
        "test": _compute_metrics(y_test, final_pipeline.predict_proba(X_test)[:, 1]),
    }
    test_metrics = metrics_by_split["test"]
    LOGGER.info(
        "Test ROC-AUC=%.3f PR-AUC=%.3f Brier=%.4f",
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["brier"],
    )

    artifact_paths = _persist_artifacts(
        out_dir=out_dir,
        pipeline=final_pipeline,
        feature_artifacts=feature_artifacts,
        model_results=model_results,
        metrics_by_split=metrics_by_split,
        split_series=split_by_id,
        seed=args.seed,
        calibration=calibration_method,
        selected_model=best_model_name,
        raw=raw,
        labels=labels,
    )

    LOGGER.info("Training complete. Artifacts written to %s", artifact_paths["model_path"])


def _build_preprocessor(artifacts: FeatureArtifacts) -> ColumnTransformer:
    transformers: List = []

    if artifacts.numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, artifacts.numeric_columns))

    if artifacts.categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, artifacts.categorical_columns))

    if not transformers:
        raise ValueError("No transformers configured; feature lists are empty.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _instantiate_model(model_name: str, seed: int):
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {model_name}") from exc
    return builder(seed)


def _compute_metrics(y_true: Iterable[int], proba: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def _subset_split(
    dataset: pd.DataFrame,
    labels: pd.Series,
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    mask = dataset["split"] == split_name
    subset = dataset.loc[mask].drop(columns=["split"])
    return subset, labels.loc[subset.index]


def _determine_calibration_folds(y_train: pd.Series) -> int:
    class_counts = y_train.value_counts()
    min_class = int(class_counts.min())
    return max(2, min(5, min_class))


def _persist_artifacts(
    out_dir: Path,
    pipeline: Pipeline,
    feature_artifacts: FeatureArtifacts,
    model_results: List[Dict[str, object]],
    metrics_by_split: Dict[str, Dict[str, float]],
    split_series: pd.Series,
    seed: int,
    calibration: str,
    selected_model: str,
    raw,
    labels: pd.Series,
) -> Dict[str, Path]:
    timestamp = datetime.utcnow().isoformat()
    model_path = out_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    split_counts = split_series.value_counts().to_dict()
    label_distribution = {
        split: labels.loc[split_series == split].value_counts().to_dict()
        for split in ("train", "val", "test")
    }

    boundaries = compute_split_boundaries(raw.advances)
    summary = {
        "generated_at": timestamp,
        "seed": seed,
        "selected_model": selected_model,
        "calibration": calibration,
        "metrics": metrics_by_split,
        "model_comparison": model_results,
        "split_counts": split_counts,
        "label_distribution": label_distribution,
        "feature_metadata": {
            "numeric_columns": feature_artifacts.numeric_columns,
            "categorical_columns": feature_artifacts.categorical_columns,
            **feature_artifacts.metadata,
        },
        "split_boundaries": {
            "start_month": str(boundaries.start_month),
            "train_end_month": str(boundaries.train_end_month),
            "val_end_month": str(boundaries.val_end_month),
            "test_end_month": str(boundaries.test_end_month),
        },
    }

    summary_path = out_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"model_path": model_path, "summary_path": summary_path}


if __name__ == "__main__":
    main()
