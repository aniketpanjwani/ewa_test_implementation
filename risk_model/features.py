"""Feature engineering utilities for the repayment risk model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .datasets import RawDataset


@dataclass
class FeatureArtifacts:
    """Metadata describing the generated feature matrix."""

    numeric_columns: List[str]
    categorical_columns: List[str]
    metadata: Dict[str, Any]


def build_feature_matrix(
    raw: RawDataset,
    cashflow_windows: Sequence[int] = (14, 30, 60, 90),
) -> Tuple[pd.DataFrame, pd.Series, FeatureArtifacts]:
    """Construct the feature matrix and target vector from the raw dataset.

    Parameters
    ----------
    raw
        Loaded dataset containing employers, users, transactions, and advances.
    cashflow_windows
        Rolling windows (in days) for cashflow aggregation.

    Returns
    -------
    features : pd.DataFrame
        Feature matrix indexed by ``advance_id``.
    labels : pd.Series
        Binary target aligned with the feature matrix (1 -> bad outcome).
    artifacts : FeatureArtifacts
        Column metadata and configuration details for downstream processing.
    """

    advances = raw.advances.copy()
    transactions = raw.transactions.copy()
    users = raw.users.set_index("user_id").copy()

    advances["bad_outcome"] = _derive_bad_outcome(advances)
    advances = advances.sort_values("requested_at").reset_index(drop=True)

    base = advances.set_index("advance_id")
    labels = base["bad_outcome"].astype(int).rename("bad_outcome")

    features = pd.DataFrame(index=base.index)
    features["user_id"] = base["user_id"]
    features["requested_at"] = base["requested_at"]
    features["advance_amount_cents"] = base["amount_cents"].astype(float)
    user_features = users[
        ["pay_frequency", "base_salary_monthly_cents", "kyc_verified"]
    ].rename(
        columns={
            "base_salary_monthly_cents": "user_base_salary_cents",
            "kyc_verified": "user_kyc_verified",
        }
    )
    features = features.join(user_features, on="user_id")

    features["user_base_salary_cents"] = features["user_base_salary_cents"].astype(float)
    features["user_kyc_verified"] = (
        features["user_kyc_verified"].astype("boolean").astype(float)
    )

    payroll_txns = transactions[transactions["category"] == "payroll"].copy()
    income_features = _compute_income_cadence_features(advances, payroll_txns)
    cashflow_features = _compute_cashflow_features(advances, transactions, cashflow_windows)
    behavioural_features = _compute_behavioural_features(advances)

    features = features.join(income_features)
    features = features.join(cashflow_features)
    features = features.join(behavioural_features)

    features["amount_to_salary_ratio"] = _safe_divide(
        features["advance_amount_cents"], features["user_base_salary_cents"]
    )

    numeric_columns = [
        col
        for col in [
            "advance_amount_cents",
            "user_base_salary_cents",
            "user_kyc_verified",
            "utilization_ratio",
            "days_since_last_payroll",
            "avg_payroll_amount_3",
            "payroll_cycle_median_days",
            "amount_to_salary_ratio",
            "last_payroll_amount",
            "prior_advances_total",
            "prior_advances_30d",
            "prior_days_since_last_advance",
            "prior_bad_rate",
            "prior_late_rate",
            "cf_credit_30d",
            "cf_debit_30d",
            "cf_txn_count_30d",
            "cf_volatility_30d",
        ]
        if col in features.columns
    ]
    numeric_columns.extend(
        col for col in _net_cashflow_columns(cashflow_windows) if col in features.columns
    )
    numeric_columns.extend(
        col for col in ["cf_ewm_net_7d", "cf_ewm_net_14d", "cf_ewm_net_30d"] if col in features.columns
    )

    categorical_columns = ["pay_frequency"]

    metadata = {
        "cashflow_windows": list(cashflow_windows),
        "feature_columns": features.columns.tolist(),
        "reference_columns": ["user_id", "requested_at"],
        "label": "bad_outcome",
    }

    artifacts = FeatureArtifacts(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        metadata=metadata,
    )

    return features, labels, artifacts


def _derive_bad_outcome(advances: pd.DataFrame) -> pd.Series:
    repaid = advances["repaid_at"]
    due = advances["due_date"]
    late_and_overdue = (
        advances["was_late"].fillna(False)
        & repaid.notna()
        & ((repaid - due) > pd.Timedelta(days=7))
    )
    wrote_off = advances["wrote_off"].fillna(False)
    return (wrote_off | late_and_overdue).astype(bool)


def _compute_income_cadence_features(
    advances: pd.DataFrame,
    payroll_txns: pd.DataFrame,
) -> pd.DataFrame:
    payroll_by_user = {
        user_id: df.sort_values("posted_at").reset_index(drop=True)
        for user_id, df in payroll_txns.groupby("user_id")
    }

    records: List[Dict[str, Any]] = []
    for user_id, user_advances in advances.groupby("user_id"):
        payroll = payroll_by_user.get(user_id)
        for row in user_advances.itertuples():
            record = {"advance_id": row.advance_id}
            if payroll is None:
                record.update(
                    {
                        "utilization_ratio": np.nan,
                        "days_since_last_payroll": np.nan,
                        "last_payroll_amount": np.nan,
                        "avg_payroll_amount_3": np.nan,
                        "payroll_cycle_median_days": np.nan,
                    }
                )
                records.append(record)
                continue

            prior = payroll[payroll["posted_at"] < row.requested_at]
            if prior.empty:
                record.update(
                    {
                        "utilization_ratio": np.nan,
                        "days_since_last_payroll": np.nan,
                        "last_payroll_amount": np.nan,
                        "avg_payroll_amount_3": np.nan,
                        "payroll_cycle_median_days": np.nan,
                    }
                )
            else:
                last_pay = prior.iloc[-1]
                delta = row.requested_at - last_pay["posted_at"]
                record["days_since_last_payroll"] = delta.total_seconds() / 86400.0
                record["last_payroll_amount"] = float(last_pay["amount_cents"])
                record["avg_payroll_amount_3"] = float(prior["amount_cents"].tail(3).mean())
                intervals = prior["posted_at"].diff().dropna()
                record["payroll_cycle_median_days"] = (
                    intervals.dt.total_seconds() / 86400.0
                ).median()
                record["utilization_ratio"] = _safe_divide(
                    float(row.amount_cents), float(last_pay["amount_cents"])
                )
            records.append(record)

    income_df = pd.DataFrame.from_records(records).set_index("advance_id")
    return income_df.astype(float)


def _compute_cashflow_features(
    advances: pd.DataFrame,
    transactions: pd.DataFrame,
    cashflow_windows: Sequence[int],
) -> pd.DataFrame:
    txns_by_user = {
        user_id: df.sort_values("posted_at").reset_index(drop=True)
        for user_id, df in transactions.groupby("user_id")
    }
    records: List[Dict[str, Any]] = []
    for user_id, user_advances in advances.groupby("user_id"):
        user_txns = txns_by_user.get(user_id)
        for row in user_advances.itertuples():
            record: Dict[str, Any] = {"advance_id": row.advance_id}
            if user_txns is None:
                for window in cashflow_windows:
                    record[f"cf_net_{window}d"] = 0.0
                record["cf_credit_30d"] = 0.0
                record["cf_debit_30d"] = 0.0
                record["cf_txn_count_30d"] = 0.0
                record["cf_volatility_30d"] = np.nan
                records.append(record)
                continue

            prior_txns = user_txns[user_txns["posted_at"] < row.requested_at]
            for window in cashflow_windows:
                start = row.requested_at - pd.Timedelta(days=window)
                window_txns = prior_txns[prior_txns["posted_at"] >= start]
                record[f"cf_net_{window}d"] = float(window_txns["amount_cents"].sum())

            window30 = prior_txns[
                prior_txns["posted_at"] >= row.requested_at - pd.Timedelta(days=30)
            ]
            record["cf_credit_30d"] = float(window30.loc[window30["amount_cents"] > 0, "amount_cents"].sum())
            record["cf_debit_30d"] = float(
                -window30.loc[window30["amount_cents"] < 0, "amount_cents"].sum()
            )
            record["cf_txn_count_30d"] = float(len(window30))

            daily_net = (
                prior_txns.set_index("posted_at")["amount_cents"]
                .resample("D")
                .sum()
            )
            if not daily_net.empty:
                daily_net = daily_net.tz_localize(None)
                start_30 = (row.requested_at - pd.Timedelta(days=30)).tz_convert(None).normalize()
                requested_day = row.requested_at.tz_convert(None).normalize()
                recent_net = daily_net.loc[start_30:requested_day]
                record["cf_volatility_30d"] = float(recent_net.std(ddof=0)) if not recent_net.empty else np.nan

                full_index = pd.date_range(daily_net.index.min(), requested_day, freq="D")
                daily_net = daily_net.reindex(full_index, fill_value=0.0)
                record["cf_ewm_net_7d"] = float(daily_net.ewm(span=7, adjust=False).mean().iloc[-1])
                record["cf_ewm_net_14d"] = float(daily_net.ewm(span=14, adjust=False).mean().iloc[-1])
                record["cf_ewm_net_30d"] = float(daily_net.ewm(span=30, adjust=False).mean().iloc[-1])
            else:
                record["cf_volatility_30d"] = np.nan
                record["cf_ewm_net_7d"] = 0.0
                record["cf_ewm_net_14d"] = 0.0
                record["cf_ewm_net_30d"] = 0.0

            records.append(record)

    cashflow_df = pd.DataFrame.from_records(records).set_index("advance_id")
    for column in cashflow_df.columns:
        if cashflow_df[column].dtype == "object":
            cashflow_df[column] = cashflow_df[column].astype(float)
    return cashflow_df


def _compute_behavioural_features(advances: pd.DataFrame) -> pd.DataFrame:
    advances = advances.sort_values(["user_id", "requested_at"]).reset_index(drop=True)
    records: List[Dict[str, Any]] = []

    for user_id, user_advances in advances.groupby("user_id"):
        user_advances = user_advances.sort_values("requested_at").reset_index(drop=True)
        for idx, row in user_advances.iterrows():
            prior = user_advances.iloc[:idx]
            record: Dict[str, Any] = {"advance_id": row["advance_id"]}
            if prior.empty:
                record.update(
                    {
                        "prior_advances_total": 0.0,
                        "prior_advances_30d": 0.0,
                        "prior_days_since_last_advance": np.nan,
                        "prior_bad_rate": np.nan,
                        "prior_late_rate": np.nan,
                    }
                )
            else:
                prior_30d = prior[
                    prior["requested_at"] >= row["requested_at"] - pd.Timedelta(days=30)
                ]
                completed = prior[prior["due_date"] < row["requested_at"]]
                record["prior_advances_total"] = float(len(prior))
                record["prior_advances_30d"] = float(len(prior_30d))
                last_request = prior["requested_at"].max()
                record["prior_days_since_last_advance"] = (
                    row["requested_at"] - last_request
                ).total_seconds() / 86400.0
                if not completed.empty:
                    record["prior_bad_rate"] = completed["bad_outcome"].astype(float).mean()
                    record["prior_late_rate"] = completed["was_late"].astype(float).mean()
                else:
                    record["prior_bad_rate"] = np.nan
                    record["prior_late_rate"] = np.nan
            records.append(record)

    behaviour_df = pd.DataFrame.from_records(records).set_index("advance_id")
    for column in behaviour_df.columns:
        behaviour_df[column] = behaviour_df[column].astype(float)
    return behaviour_df


def _net_cashflow_columns(cashflow_windows: Sequence[int]) -> List[str]:
    return [f"cf_net_{window}d" for window in cashflow_windows]


def _safe_divide(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
) -> pd.Series | float:
    result = numerator / denominator
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    if not np.isfinite(result):
        return np.nan
    return result


__all__ = ["FeatureArtifacts", "build_feature_matrix"]
