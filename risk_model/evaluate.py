"""Evaluation entrypoint for the repayment risk model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the repayment risk model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the generated dataset directory.")
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory containing trained model artifacts.")
    parser.add_argument("--out", type=Path, default=Path("./artifacts/risk"), help="Directory for evaluation artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise NotImplementedError("Model evaluation routine has not been implemented yet.")


if __name__ == "__main__":
    main()
