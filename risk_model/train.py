"""Training entrypoint for the repayment risk model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the repayment risk model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the generated dataset directory.")
    parser.add_argument("--out", type=Path, required=True, help="Directory where model artifacts will be written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise NotImplementedError("Model training routine has not been implemented yet.")


if __name__ == "__main__":
    main()
