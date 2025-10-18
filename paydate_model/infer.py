"""User-facing interface for next-payday prediction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from .predictor import PaydatePredictor, PredictionResult


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _project_root() / "data"


def _schedule_path() -> Path:
    return _project_root() / "artifacts" / "paydate" / "schedules.parquet"


@lru_cache(maxsize=1)
def _get_predictor() -> PaydatePredictor:
    return PaydatePredictor(data_dir=_data_dir(), schedule_path=_schedule_path())


def predict_next(user_id: str, reference_date: Optional[str] = None) -> dict:
    predictor = _get_predictor()
    result: PredictionResult = predictor.predict_user(user_id, reference_date=reference_date)
    return result.to_dict()


__all__ = ["predict_next"]
