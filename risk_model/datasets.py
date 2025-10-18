"""Data ingestion utilities for the repayment risk model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

RAW_FILENAMES: Dict[str, str] = {
    "employers": "employers.csv",
    "users": "users.csv",
    "transactions": "transactions.csv",
    "advances": "advances.csv",
}

EXPECTED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "employers": ("employer_id", "name", "industry", "size"),
    "users": (
        "user_id",
        "employer_id",
        "hire_date",
        "base_salary_monthly_cents",
        "pay_frequency",
        "kyc_verified",
    ),
    "transactions": (
        "user_id",
        "txn_id",
        "posted_at",
        "amount_cents",
        "merchant",
        "category",
        "type",
    ),
    "advances": (
        "advance_id",
        "user_id",
        "requested_at",
        "amount_cents",
        "due_date",
        "repaid_at",
        "was_late",
        "wrote_off",
    ),
}


@dataclass
class RawDataset:
    """Container for the raw input tables."""

    employers: pd.DataFrame
    users: pd.DataFrame
    transactions: pd.DataFrame
    advances: pd.DataFrame

    def summarize(self) -> Dict[str, int]:
        """Return the row counts for each table."""
        return {
            "employers": len(self.employers),
            "users": len(self.users),
            "transactions": len(self.transactions),
            "advances": len(self.advances),
        }


def _validate_columns(table_name: str, df: pd.DataFrame) -> None:
    """Ensure the dataframe contains at least the expected columns."""
    expected = set(EXPECTED_COLUMNS[table_name])
    missing = expected.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{table_name} is missing columns: {missing_cols}")


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")


def load_raw_dataset(data_dir: Path) -> RawDataset:
    """Load and validate the raw CSV exports from the generator script.

    Parameters
    ----------
    data_dir
        Directory containing the four CSV outputs produced by generate_ewa_synth.py.

    Returns
    -------
    RawDataset
        Dataclass containing cleaned pandas DataFrames for downstream processing.
    """

    data_dir = data_dir.expanduser().resolve()
    employers, users, transactions, advances = (
        _load_employers(data_dir),
        _load_users(data_dir),
        _load_transactions(data_dir),
        _load_advances(data_dir),
    )
    return RawDataset(
        employers=employers,
        users=users,
        transactions=transactions,
        advances=advances,
    )


def _resolve(data_dir: Path, key: str) -> Path:
    filename = RAW_FILENAMES[key]
    path = data_dir / filename
    _ensure_exists(path)
    return path


def _load_employers(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(data_dir, "employers"))
    _validate_columns("employers", df)
    return df


def _load_users(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(data_dir, "users"))
    _validate_columns("users", df)
    df["hire_date"] = pd.to_datetime(df["hire_date"], utc=True)
    df["kyc_verified"] = (
        df["kyc_verified"].astype(str).str.lower().map({"true": True, "false": False}).astype("boolean")
    )
    return df


def _load_transactions(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(data_dir, "transactions"))
    _validate_columns("transactions", df)
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True)
    df = df.sort_values(["user_id", "posted_at", "txn_id"], ignore_index=True)
    return df


def _load_advances(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(data_dir, "advances"))
    _validate_columns("advances", df)
    df["requested_at"] = pd.to_datetime(df["requested_at"], utc=True)
    df["due_date"] = pd.to_datetime(df["due_date"], utc=True)
    df["repaid_at"] = pd.to_datetime(df["repaid_at"], utc=True, errors="coerce")
    df["was_late"] = (
        df["was_late"].astype(str).str.lower().map({"true": True, "false": False}).astype("boolean")
    )
    df["wrote_off"] = (
        df["wrote_off"].astype(str).str.lower().map({"true": True, "false": False}).astype("boolean")
    )
    df = df.sort_values(["requested_at", "advance_id"], ignore_index=True)
    return df


def month_period(series: pd.Series) -> pd.Series:
    """Utility to convert timestamps to month periods (UTC aware)."""
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series.dt.to_period("M")


__all__ = ["RawDataset", "load_raw_dataset", "month_period"]
