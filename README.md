# EWA Assessment Implementation

This repository houses the working implementation for the EWA take-home assessment.
It follows the structure recommended in the challenge brief and now contains end-to-end
pipelines for the repayment risk model (Task 2A) and an in-progress pay-date predictor
(Task 2B).

## Task status (per `../data-challenge/README.md`)

- **2A — Repayment Risk Model (complete):**
  - LightGBM + isotonic calibration achieves ROC‑AUC 0.701, PR‑AUC 0.314, Brier 0.146, and ECE 0.016 on the held-out 3-month window.
  - Time-aware splits, leakage guards, slice analysis, business simulation, and unit tests are implemented (`risk_model/report.md`, `tests/test_risk.py`).
- **2B — Pay-Date Prediction (in progress):**
  - Heuristic predictor with cadence inference and per-user monthly offsets now delivers MAE 1.85 days overall (monthly MAE 1.15) and ±2-day hit rate 69.6% (monthly 82.1%).
  - Further tuning is needed to hit the overall MAE ≤ 1.8 and ±2-day ≥ 80% acceptance criteria; see `docs/paydate_model/report.md` for roadmap.
- **2C — Support Bot:** not yet implemented (skeleton only).

## Repository layout (current)

- `artifacts/`: experiment outputs (risk metrics, paydate schedules/eval tables).
- `data/`: synthetic dataset generated with `generate_ewa_synth.py`.
- `docs/`: assessment documentation (FAQ corpus, reports, task plans).
- `paydate_model/`: cadence inference, predictor, CLI evaluation code.
- `risk_model/`: feature engineering, training/tuning scripts, evaluation.
- `support_bot/`: placeholder for future FAQ bot implementation.
- `tests/`: pytest suites (`test_risk.py`, `test_paydate.py`).
- `pyproject.toml` / `uv.lock`: dependency definitions managed with `uv`.
- `Makefile`: mirrors assessment targets (`risk-train`, `risk-eval`, `paydate-eval`, etc.).

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

## Support bot (Task 2C) outlook

The challenge expects a grounded FAQ assistant that answers user questions using
local documentation. The proposed approach keeps everything self-contained:
Markdown help files would live under `implementation/docs`, we would index those
passages with BM25 for lexical matching, and optionally rerank the top hits with
lightweight sentence embeddings so the most relevant snippets surface first. Answers
could then be constructed by extracting or lightly compressing the best matching
passage and always appending explicit citations (for example, `Source: fees.md`)
to satisfy the grounding requirement.

Evaluation would replay the provided golden questions by calling the same CLI,
measure semantic similarity against reference answers, confirm that cited sources
match the expected filenames, and track latency to stay within the 500 ms budget.
Metrics and diagnostics would be written alongside the bot artifacts so reviewers can
audit retrieval quality without rerunning the pipeline.

Today, this repository cannot implement or validate that plan because the FAQ
Markdown corpus and `bot_eval.jsonl` golden set are not included in the
`free-doom/data-challenge` repository. Once those assets arrive, populate
`implementation/docs`, build the BM25/embedding artifacts, and wire up the CLI plus
evaluation harness following the outline above.
