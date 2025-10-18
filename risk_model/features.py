"""Feature engineering utilities for the repayment risk model."""

from __future__ import annotations

from collections import deque
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
        ["pay_frequency", "base_salary_monthly_cents", "kyc_verified", "hire_date"]
    ].rename(
        columns={
            "base_salary_monthly_cents": "user_base_salary_cents",
            "kyc_verified": "user_kyc_verified",
        }
    )
    features = features.join(user_features, on="user_id")
    features["employer_id"] = features["user_id"].map(users["employer_id"]).astype("string")

    features["user_base_salary_cents"] = features["user_base_salary_cents"].astype(float)
    features["user_kyc_verified"] = (
        features["user_kyc_verified"].astype("boolean").astype(float)
    )
    requested_ts = pd.to_datetime(features["requested_at"], utc=True)
    hire_dates = pd.to_datetime(features["hire_date"], utc=True)
    features["user_tenure_days"] = (
        (requested_ts - hire_dates).dt.total_seconds() / 86400.0
    )
    features = features.drop(columns=["hire_date"])

    payroll_txns = transactions[transactions["category"] == "payroll"].copy()
    income_features = _compute_income_cadence_features(advances, payroll_txns)
    cashflow_features = _compute_cashflow_features(advances, transactions, cashflow_windows)
    behavioural_features = _compute_behavioural_features(advances)
    employer_features = _compute_employer_features(
        advances, users, income_features.get("utilization_ratio")
    )

    features = features.join(income_features)
    features = features.join(cashflow_features)
    features = features.join(behavioural_features)
    features = features.join(employer_features)

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
            "payroll_amount_std_3",
            "payroll_amount_cov_3",
            "payroll_amount_std_6",
            "payroll_amount_cov_6",
            "last_payroll_to_median_6",
            "payroll_cycle_delta",
            "user_util_mean",
            "user_util_median",
            "user_util_std",
            "user_util_last6_mean",
            "user_util_last6_std",
            "user_util_last6_max",
            "user_util_last6_min",
            "user_util_deviation_median",
            "user_util_zscore",
            "amount_to_salary_ratio",
            "last_payroll_amount",
            "user_tenure_days",
            "prior_advances_total",
            "prior_advances_30d",
            "prior_days_since_last_advance",
            "prior_bad_rate",
            "prior_late_rate",
            "prior_bad_streak",
            "prior_late_streak",
            "prior_writeoff_streak",
            "prior_days_since_last_late",
            "prior_days_since_last_writeoff",
            "prior_bad_events_total",
            "prior_late_events_total",
            "prior_writeoff_events_total",
            "employer_prior_adv_count",
            "employer_prior_bad_rate",
            "employer_prior_late_rate",
            "employer_prior_writeoff_rate",
            "employer_prior_bad_rate_30d",
            "employer_prior_adv_30d",
            "employer_prior_user_diversity",
            "employer_prior_adv_per_user",
            "employer_prior_util_mean",
            "employer_prior_util_std",
            "employer_prior_util_last6_mean",
            "employer_prior_util_last6_std",
            "employer_prior_util_deviation",
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
    if "employer_id" in features.columns:
        categorical_columns.append("employer_id")

    metadata = {
        "cashflow_windows": list(cashflow_windows),
        "feature_columns": features.columns.tolist(),
        "reference_columns": ["user_id", "requested_at", "employer_id"],
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

    default_fields = {
        "utilization_ratio": np.nan,
        "days_since_last_payroll": np.nan,
        "last_payroll_amount": np.nan,
        "avg_payroll_amount_3": np.nan,
        "payroll_cycle_median_days": np.nan,
        "payroll_amount_std_3": np.nan,
        "payroll_amount_cov_3": np.nan,
        "payroll_amount_std_6": np.nan,
        "payroll_amount_cov_6": np.nan,
        "last_payroll_to_median_6": np.nan,
        "payroll_cycle_delta": np.nan,
        "user_util_mean": np.nan,
        "user_util_median": np.nan,
        "user_util_std": np.nan,
        "user_util_last6_mean": np.nan,
        "user_util_last6_std": np.nan,
        "user_util_last6_max": np.nan,
        "user_util_last6_min": np.nan,
        "user_util_deviation_median": np.nan,
        "user_util_zscore": np.nan,
    }

    records: List[Dict[str, Any]] = []
    for user_id, user_advances in advances.groupby("user_id"):
        payroll = payroll_by_user.get(user_id)
        util_history: List[float] = []
        recent_util_history: deque[float] = deque(maxlen=6)

        for row in user_advances.itertuples():
            record: Dict[str, Any] = {"advance_id": row.advance_id}
            record.update(default_fields)

            if payroll is None:
                records.append(record)
                continue

            prior = payroll[payroll["posted_at"] < row.requested_at]
            if prior.empty:
                records.append(record)
                continue

            last_pay = prior.iloc[-1]
            delta = row.requested_at - last_pay["posted_at"]
            record["days_since_last_payroll"] = delta.total_seconds() / 86400.0
            record["last_payroll_amount"] = float(last_pay["amount_cents"])
            record["avg_payroll_amount_3"] = float(prior["amount_cents"].tail(3).mean())

            intervals = prior["posted_at"].diff().dropna()
            record["payroll_cycle_median_days"] = (
                intervals.dt.total_seconds() / 86400.0
            ).median()

            util_value = _safe_divide(float(row.amount_cents), float(last_pay["amount_cents"]))
            record["utilization_ratio"] = util_value

            trailing3 = prior["amount_cents"].tail(3)
            trailing6 = prior["amount_cents"].tail(6)
            std3 = float(trailing3.std(ddof=0)) if not trailing3.empty else np.nan
            mean3 = float(trailing3.mean()) if not trailing3.empty else np.nan
            cov3 = np.nan
            if not np.isnan(std3) and not np.isnan(mean3) and mean3 != 0.0:
                cov3 = std3 / mean3

            std6 = float(trailing6.std(ddof=0)) if not trailing6.empty else np.nan
            mean6 = float(trailing6.mean()) if not trailing6.empty else np.nan
            cov6 = np.nan
            if not np.isnan(std6) and not np.isnan(mean6) and mean6 != 0.0:
                cov6 = std6 / mean6

            median6 = float(trailing6.median()) if not trailing6.empty else np.nan

            record["payroll_amount_std_3"] = std3
            record["payroll_amount_cov_3"] = cov3
            record["payroll_amount_std_6"] = std6
            record["payroll_amount_cov_6"] = cov6
            record["last_payroll_to_median_6"] = (
                _safe_divide(record["last_payroll_amount"], median6) if not np.isnan(median6) else np.nan
            )

            days = record["days_since_last_payroll"]
            cycle = record["payroll_cycle_median_days"]
            if not np.isnan(days) and not np.isnan(cycle):
                record["payroll_cycle_delta"] = abs(days - cycle)

            valid_history = [u for u in util_history if not np.isnan(u)]
            if valid_history:
                arr = np.array(valid_history, dtype=float)
                record["user_util_mean"] = float(arr.mean())
                record["user_util_median"] = float(np.median(arr))
                record["user_util_std"] = float(arr.std(ddof=0))

            recent_valid = [u for u in recent_util_history if not np.isnan(u)]
            if recent_valid:
                recent_arr = np.array(recent_valid, dtype=float)
                record["user_util_last6_mean"] = float(recent_arr.mean())
                record["user_util_last6_std"] = float(recent_arr.std(ddof=0))
                record["user_util_last6_max"] = float(recent_arr.max())
                record["user_util_last6_min"] = float(recent_arr.min())

            median_val = record["user_util_median"]
            if not np.isnan(util_value) and not np.isnan(median_val):
                record["user_util_deviation_median"] = float(util_value - median_val)

            mean_val = record["user_util_mean"]
            std_val = record["user_util_std"]
            if (
                not np.isnan(util_value)
                and not np.isnan(mean_val)
                and not np.isnan(std_val)
                and std_val > 0
            ):
                record["user_util_zscore"] = float((util_value - mean_val) / std_val)

            if not np.isnan(util_value):
                util_history.append(float(util_value))
                recent_util_history.append(float(util_value))
            else:
                util_history.append(np.nan)
                recent_util_history.append(np.nan)

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
        bad_streak = 0.0
        late_streak = 0.0
        writeoff_streak = 0.0
        last_late_ts: pd.Timestamp | None = None
        last_writeoff_ts: pd.Timestamp | None = None
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
                        "prior_bad_streak": 0.0,
                        "prior_late_streak": 0.0,
                        "prior_writeoff_streak": 0.0,
                        "prior_days_since_last_late": np.nan,
                        "prior_days_since_last_writeoff": np.nan,
                        "prior_bad_events_total": 0.0,
                        "prior_late_events_total": 0.0,
                        "prior_writeoff_events_total": 0.0,
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

                record["prior_bad_events_total"] = float(prior["bad_outcome"].astype(float).sum())
                record["prior_late_events_total"] = float(prior["was_late"].astype(float).sum())
                record["prior_writeoff_events_total"] = float(prior["wrote_off"].astype(float).sum())

                record["prior_bad_streak"] = bad_streak
                record["prior_late_streak"] = late_streak
                record["prior_writeoff_streak"] = writeoff_streak
                record["prior_days_since_last_late"] = (
                    (row["requested_at"] - last_late_ts).total_seconds() / 86400.0
                    if last_late_ts is not None
                    else np.nan
                )
                record["prior_days_since_last_writeoff"] = (
                    (row["requested_at"] - last_writeoff_ts).total_seconds() / 86400.0
                    if last_writeoff_ts is not None
                    else np.nan
                )
            if prior.empty:
                record["prior_bad_events_total"] = 0.0
                record["prior_late_events_total"] = 0.0
                record["prior_writeoff_events_total"] = 0.0
            records.append(record)

            # Update streak trackers with current outcome
            if bool(row["bad_outcome"]):
                bad_streak += 1.0
            else:
                bad_streak = 0.0

            if bool(row["was_late"]):
                late_streak += 1.0
                last_late_ts = row["requested_at"]
            else:
                late_streak = 0.0

            if bool(row["wrote_off"]):
                writeoff_streak += 1.0
                last_writeoff_ts = row["requested_at"]
            else:
                writeoff_streak = 0.0

    behaviour_df = pd.DataFrame.from_records(records).set_index("advance_id")
    for column in behaviour_df.columns:
        behaviour_df[column] = behaviour_df[column].astype(float)
    return behaviour_df


def _compute_employer_features(
    advances: pd.DataFrame,
    users: pd.DataFrame,
    utilization: pd.Series | None = None,
) -> pd.DataFrame:
    user_employers = users["employer_id"].dropna()
    employer_records: List[Dict[str, Any]] = []

    adv_emp = advances.copy()
    adv_emp["employer_id"] = adv_emp["user_id"].map(user_employers)
    adv_emp = adv_emp.dropna(subset=["employer_id"])
    adv_emp = adv_emp.sort_values("requested_at")
    if utilization is not None:
        adv_emp = adv_emp.merge(
            utilization.rename("utilization_ratio"),
            left_on="advance_id",
            right_index=True,
            how="left",
        )
    else:
        adv_emp["utilization_ratio"] = np.nan

    default_fields = {
        "employer_prior_adv_count": 0.0,
        "employer_prior_bad_rate": np.nan,
        "employer_prior_late_rate": np.nan,
        "employer_prior_writeoff_rate": np.nan,
        "employer_prior_bad_rate_30d": np.nan,
        "employer_prior_adv_30d": 0.0,
        "employer_prior_user_diversity": 0.0,
        "employer_prior_adv_per_user": np.nan,
        "employer_prior_util_mean": np.nan,
        "employer_prior_util_std": np.nan,
        "employer_prior_util_last6_mean": np.nan,
        "employer_prior_util_last6_std": np.nan,
        "employer_prior_util_deviation": np.nan,
    }

    for employer_id, group in adv_emp.groupby("employer_id"):
        group = group.sort_values("requested_at").reset_index(drop=True)
        for idx, row in group.iterrows():
            prior = group.iloc[:idx]
            record: Dict[str, Any] = {"advance_id": row["advance_id"]}
            record.update(default_fields)
            if not prior.empty:
                prior_30d = prior[
                    prior["requested_at"] >= row["requested_at"] - pd.Timedelta(days=30)
                ]
                record["employer_prior_adv_count"] = float(len(prior))
                record["employer_prior_bad_rate"] = prior["bad_outcome"].astype(float).mean()
                record["employer_prior_late_rate"] = prior["was_late"].astype(float).mean()
                record["employer_prior_writeoff_rate"] = prior["wrote_off"].astype(float).mean()
                record["employer_prior_bad_rate_30d"] = (
                    prior_30d["bad_outcome"].astype(float).mean() if not prior_30d.empty else np.nan
                )
                record["employer_prior_adv_30d"] = float(len(prior_30d))
                unique_users = prior["user_id"].nunique()
                record["employer_prior_user_diversity"] = float(unique_users)
                record["employer_prior_adv_per_user"] = (
                    float(len(prior)) / unique_users if unique_users else np.nan
                )

                prior_util = prior["utilization_ratio"].to_numpy(dtype=float)
                prior_util = prior_util[~np.isnan(prior_util)]
                if prior_util.size > 0:
                    record["employer_prior_util_mean"] = float(prior_util.mean())
                    record["employer_prior_util_std"] = float(prior_util.std(ddof=0))

                recent_prior = prior.tail(6)["utilization_ratio"].to_numpy(dtype=float)
                recent_prior = recent_prior[~np.isnan(recent_prior)]
                if recent_prior.size > 0:
                    record["employer_prior_util_last6_mean"] = float(recent_prior.mean())
                    record["employer_prior_util_last6_std"] = float(recent_prior.std(ddof=0))

                current_util = row.get("utilization_ratio", np.nan)
                mean_val = record.get("employer_prior_util_mean", np.nan)
                if not np.isnan(current_util) and not np.isnan(mean_val):
                    record["employer_prior_util_deviation"] = float(current_util - mean_val)
                else:
                    record["employer_prior_util_deviation"] = np.nan

            current_util_val = row.get("utilization_ratio", np.nan)
            employer_records.append(record)

    employer_df = pd.DataFrame.from_records(employer_records).set_index("advance_id")
    for column in employer_df.columns:
        employer_df[column] = employer_df[column].astype(float)
    return employer_df


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
