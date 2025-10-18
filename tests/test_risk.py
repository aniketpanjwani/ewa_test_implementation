from __future__ import annotations

import pandas as pd
import pytest

from risk_model.datasets import RawDataset
from risk_model.features import build_feature_matrix


def _raw_dataset(
    employers: pd.DataFrame,
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    advances: pd.DataFrame,
) -> RawDataset:
    return RawDataset(
        employers=employers.reset_index(drop=True),
        users=users.reset_index(drop=True),
        transactions=transactions.reset_index(drop=True),
        advances=advances.reset_index(drop=True),
    )


def test_cashflow_features_exclude_post_request_transactions():
    """Ensure rolling cashflow features do not leak post-request activity."""
    ts = pd.Timestamp
    employers = pd.DataFrame({"employer_id": ["e1"], "name": ["Acme"], "industry": ["tech"], "size": [50]})
    users = pd.DataFrame(
        {
            "user_id": ["u1"],
            "employer_id": ["e1"],
            "hire_date": [ts("2023-01-01T00:00:00Z")],
            "base_salary_monthly_cents": [500_000],
            "pay_frequency": ["biweekly"],
            "kyc_verified": [True],
        }
    )
    transactions = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "txn_id": "t_before",
                "posted_at": ts("2024-01-01T08:00:00Z"),
                "amount_cents": 200_00,
                "merchant": "PayrollCo",
                "category": "payroll",
                "type": "credit",
            },
            {
                "user_id": "u1",
                "txn_id": "t_after",
                "posted_at": ts("2024-01-10T08:00:00Z"),
                "amount_cents": 1_000_00,
                "merchant": "BonusCorp",
                "category": "payroll",
                "type": "credit",
            },
        ]
    )
    advances = pd.DataFrame(
        [
            {
                "advance_id": "a1",
                "user_id": "u1",
                "requested_at": ts("2024-01-05T12:00:00Z"),
                "amount_cents": 50_00,
                "due_date": ts("2024-01-15T12:00:00Z"),
                "repaid_at": ts("2024-01-15T12:00:00Z"),
                "was_late": False,
                "wrote_off": False,
            },
        ]
    )

    raw = _raw_dataset(employers, users, transactions, advances)
    features, _, _ = build_feature_matrix(raw)
    row = features.loc["a1"]

    # Only the pre-request transaction should contribute to the rolling windows.
    assert pytest.approx(row["cf_net_30d"]) == 200_00
    assert pytest.approx(row["cf_credit_30d"]) == 200_00
    assert row["cf_net_14d"] == pytest.approx(200_00)


def test_utilization_highers_bad_rate():
    """Higher utilization ratios should correlate with higher observed bad outcomes."""
    ts = pd.Timestamp
    employers = pd.DataFrame(
        {
            "employer_id": ["e1", "e2"],
            "name": ["Acme", "Globex"],
            "industry": ["tech", "retail"],
            "size": [100, 200],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": ["u_low", "u_high"],
            "employer_id": ["e1", "e2"],
            "hire_date": [ts("2023-01-01T00:00:00Z"), ts("2023-01-01T00:00:00Z")],
            "base_salary_monthly_cents": [600_000, 300_000],
            "pay_frequency": ["biweekly", "biweekly"],
            "kyc_verified": [True, True],
        }
    )
    transactions = pd.DataFrame(
        [
            {
                "user_id": "u_low",
                "txn_id": "t_low_pay",
                "posted_at": ts("2024-01-01T08:00:00Z"),
                "amount_cents": 200_000,
                "merchant": "PayrollCo",
                "category": "payroll",
                "type": "credit",
            },
            {
                "user_id": "u_high",
                "txn_id": "t_high_pay",
                "posted_at": ts("2024-01-01T08:00:00Z"),
                "amount_cents": 50_000,
                "merchant": "PayrollCo",
                "category": "payroll",
                "type": "credit",
            },
        ]
    )
    advances = pd.DataFrame(
        [
            {
                "advance_id": "a_low",
                "user_id": "u_low",
                "requested_at": ts("2024-01-05T12:00:00Z"),
                "amount_cents": 20_000,
                "due_date": ts("2024-01-20T12:00:00Z"),
                "repaid_at": ts("2024-01-20T12:00:00Z"),
                "was_late": False,
                "wrote_off": False,
            },
            {
                "advance_id": "a_high",
                "user_id": "u_high",
                "requested_at": ts("2024-01-05T12:00:00Z"),
                "amount_cents": 40_000,
                "due_date": ts("2024-01-20T12:00:00Z"),
                "repaid_at": ts("2024-02-05T12:00:00Z"),
                "was_late": True,
                "wrote_off": False,
            },
        ]
    )

    raw = _raw_dataset(employers, users, transactions, advances)
    features, labels, _ = build_feature_matrix(raw)

    df = pd.DataFrame({"utilization": features["utilization_ratio"], "label": labels})
    df["bucket"] = pd.qcut(df["utilization"], q=2, labels=["low", "high"])
    grouped = df.groupby("bucket", observed=False)["label"].mean()

    assert float(grouped.loc["high"]) > float(grouped.loc["low"])
