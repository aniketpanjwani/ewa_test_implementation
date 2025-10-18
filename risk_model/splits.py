"""Time-aware dataset splitting helpers for the repayment risk model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .datasets import month_period

SplitLabel = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitBoundaries:
    """Metadata describing the temporal split configuration."""

    start_month: pd.Period
    train_end_month: pd.Period
    val_end_month: pd.Period
    test_end_month: pd.Period


def assign_time_split(
    advances: pd.DataFrame,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
) -> pd.Series:
    """Label each advance by the time-aware train/val/test split.

    Parameters
    ----------
    advances
        DataFrame containing an already-parsed ``requested_at`` timestamp column.
    train_months, val_months, test_months
        Number of consecutive months assigned to each split. Defaults match the
        challenge brief (12/3/3).

    Returns
    -------
    pd.Series
        Categorical series aligned with ``advances`` indicating the split label.
    """

    if "requested_at" not in advances.columns:
        raise KeyError("advances dataframe must include a 'requested_at' column.")

    periods = month_period(advances["requested_at"])
    start_month = periods.min()
    if pd.isna(start_month):
        raise ValueError("Unable to determine start month from empty advances dataframe.")

    offsets = _month_offsets(periods, start_month)
    train_cutoff = train_months
    val_cutoff = train_cutoff + val_months
    test_cutoff = val_cutoff + test_months

    labels = pd.Series(index=advances.index, dtype="object")
    labels[offsets < train_cutoff] = "train"
    labels[(offsets >= train_cutoff) & (offsets < val_cutoff)] = "val"
    labels[offsets >= val_cutoff] = "test"

    return labels.astype(pd.CategoricalDtype(categories=["train", "val", "test"]))


def compute_split_boundaries(
    advances: pd.DataFrame,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
) -> SplitBoundaries:
    """Return the period boundaries that define the time-aware split."""

    periods = month_period(advances["requested_at"])
    start_month = periods.min()
    if pd.isna(start_month):
        raise ValueError("Unable to compute boundaries from empty advances dataframe.")

    train_end = start_month + train_months - 1
    val_end = train_end + val_months
    test_end = val_end + test_months

    return SplitBoundaries(
        start_month=start_month,
        train_end_month=train_end,
        val_end_month=val_end,
        test_end_month=test_end,
    )


def summarize_splits(labels: pd.Series) -> pd.DataFrame:
    """Provide counts per split label as a dataframe."""
    counts = labels.value_counts().reindex(["train", "val", "test"], fill_value=0)
    return counts.rename_axis("split").reset_index(name="count")


def _month_offsets(periods: pd.Series, start_month: pd.Period) -> pd.Series:
    """Compute month offsets from the start period."""
    years = periods.dt.year - start_month.year
    months = periods.dt.month - start_month.month
    return years * 12 + months


__all__ = [
    "SplitBoundaries",
    "assign_time_split",
    "compute_split_boundaries",
    "summarize_splits",
]
