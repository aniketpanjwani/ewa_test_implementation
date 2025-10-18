from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from paydate_model import schedule
from paydate_model.predictor import PaydatePredictor


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def schedule_table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialise the schedule table once for reuse across tests."""

    tmp_dir = tmp_path_factory.mktemp("paydate_schedule")
    table = schedule.build_schedule_table(DATA_DIR)
    path = tmp_dir / "schedules.parquet"
    table.to_parquet(path, index=False)
    return path



def _history(user_id: str) -> pd.Series:
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    transactions = transactions[transactions["category"] == "payroll"].copy()
    transactions["posted_at"] = pd.to_datetime(transactions["posted_at"])
    history = transactions[transactions["user_id"] == user_id].sort_values("posted_at")
    return history["posted_at"].reset_index(drop=True)


def _reference_and_actual(user_id: str, actual_index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    history = _history(user_id)
    actual = history.iloc[actual_index]
    reference = history.iloc[actual_index - 1]
    return reference.tz_convert(None), actual.tz_convert(None)


def test_schedule_monthly_offsets_capture_jitter(schedule_table: Path) -> None:
    table = pd.read_parquet(schedule_table)
    monthly_rows = table[table["declared_frequency"] == "monthly"]
    assert not monthly_rows.empty

    # Offsets should be finite and within the jitter window introduced by the generator.
    for column in ["monthly_offset_median", "monthly_offset_q10", "monthly_offset_q90"]:
        assert monthly_rows[column].notna().all(), f"{column} contains null offsets"
        assert ((monthly_rows[column] >= -7) & (monthly_rows[column] <= 7)).all()


@pytest.mark.parametrize(
    "user_id,actual_index",
    [
        ("u_01206", 14),
        ("u_00001", 73),
        ("u_00317", 34),
    ],
)
def test_predict_next_within_two_days(schedule_table: Path, user_id: str, actual_index: int) -> None:
    predictor = PaydatePredictor(data_dir=DATA_DIR, schedule_path=schedule_table)
    reference_ts, actual_ts = _reference_and_actual(user_id, actual_index)
    prediction = predictor.predict_user(user_id, reference_date=reference_ts.isoformat())
    predicted_ts = pd.to_datetime(prediction.date)
    assert abs((predicted_ts - actual_ts).days) <= 2


def test_interval_bounds_are_consistent(schedule_table: Path) -> None:
    predictor = PaydatePredictor(data_dir=DATA_DIR, schedule_path=schedule_table)
    reference_ts, _ = _reference_and_actual("u_01206", 14)
    prediction = predictor.predict_user("u_01206", reference_date=reference_ts.isoformat())
    center = pd.to_datetime(prediction.date)
    low = pd.to_datetime(prediction.low) if prediction.low else None
    high = pd.to_datetime(prediction.high) if prediction.high else None

    assert low is not None and high is not None
    assert low <= center <= high
    assert (high - low).days <= 5  # monthly offsets are limited to small jitter bands
