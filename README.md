# EWA Assessment Implementation

This repository houses the working implementation for the EWA take-home assessment.
It follows the structure recommended in the challenge brief and currently focuses on
standing up the repayment risk model workflow. Companion documentation lives in the
root project under `../docs`.

## Repository layout (initial)

- `data/`: synthetic dataset generated with `generate_ewa_synth.py`.
- `docs/`: provided FAQ corpus for the support bot.
- `artifacts/`: model artifacts produced during experimentation.
- `risk_model/`: feature engineering, training, evaluation, and reporting assets.
- `paydate_model/`: placeholder for the next pay date model.
- `support_bot/`: placeholder for the FAQ bot.
- `tests/`: shared unit and integration tests.
- `pyproject.toml`: project metadata and dependency definitions managed with `uv`.
- `Makefile`: developer conveniences mirroring the challenge brief.

## Getting started

1. Install [uv](https://github.com/astral-sh/uv) if it is not already available.
2. Create a virtual environment and install dependencies:

   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e .[dev]
   ```

3. Use `make` targets (e.g. `make risk-train`) to execute commands inside the `uv`
   environment; the recipes call `uv run python …` to ensure isolation.
