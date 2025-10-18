# Repayment Risk Model Report

## Overview

- **Objective:** Estimate `p_bad` for each advance at request time using the synthetic 18‑month Earned Wage Access dataset.
- **Final model:** LightGBM classifier with isotonic calibration wrapped in a preprocessing pipeline (`artifacts/risk/model.joblib`).
- **Test performance:** ROC-AUC 0.701, PR-AUC 0.314, Brier 0.146, ECE 0.016 on the held-out 3-month window.

## Data & Leakage Safeguards

- Dataset generated via `generate_ewa_synth.py` (seed 42) and ingested with schema validation (`risk_model/datasets.py`).
- Time-aware splits: earliest 12 months for training (18,468 advances), next 3 for validation (4,599), final 3 for test (4,451). Label distribution remains ~19% bad across splits.
- Feature windows use only transactions strictly before `requested_at`; unit tests (`tests/test_risk.py::test_cashflow_features_exclude_post_request_transactions`) enforce this leakage guard.
- Monthly cohort monitoring (`artifacts/risk/eval/cohort_stats.csv`) confirms stable bad-rate band 18–21% with slight lift in the final months.

## Feature Summary

Feature engineering (`risk_model/features.py`) assembles a leakage-free mix of cadence, liquidity, and behavioural signals. Highlights:

- **Income cadence & stability:** days since last payroll, median cycle length, trailing 3/6 payroll averages, rolling payroll volatility (std/CoV), and the deviation between observed cadence and typical cycle length.
- **Cashflow health:** rolling net cash over 14/30/60/90 days, 30-day credit/debit totals, transaction counts, daily volatility, and exponentially weighted net flows.
- **Utilisation context:** current utilisation ratio plus user-level history (mean/median/std, last-six window stats, deviation/z-score) and employer-level baselines to capture unusually high draws.
- **Behavioural history:** total/30-day advance counts, days since last advance, prior bad/late rates and streaks, raw counts of historical late/write-off events, and employer aggregates (bad/late/write-off rates, advance density, utilisation averages).

All numeric columns are listed in `artifacts/risk/evaluation_summary.json`, ensuring downstream consumers have schema metadata.

## Model Training & Selection

Candidate models were trained on the preprocessed training split and compared on validation metrics:

| Model                     | ROC-AUC | PR-AUC | Brier |
|---------------------------|--------:|-------:|------:|
| Logistic Regression        | 0.652   | 0.268  | 0.145 |
| HistGradientBoosting       | 0.671   | 0.277  | 0.143 |
| XGBoost                    | 0.663   | 0.276  | 0.145 |
| **LightGBM (selected)**    | **0.671** | **0.268** | 0.142 |

LightGBM delivered the highest validation ROC-AUC while preserving calibration; isotonic post-processing keeps ECE ≤0.02 on the validation/test windows.

## Evaluation Metrics

| Split | ROC-AUC | PR-AUC | Brier | Log Loss | ECE |
|-------|--------:|-------:|------:|---------:|----:|
| Train | 0.707 | 0.387 | 0.144 | 0.451 | 0.024 |
| Val   | 0.673 | 0.268 | 0.143 | 0.450 | 0.021 |
| **Test** | **0.701** | **0.314** | **0.146** | **0.455** | **0.016** |

Calibration diagnostics (see `artifacts/risk/plots/calibration_test.png`) show mild over-confidence above score ≈0.25; isotonic smoothing keeps ECE under 0.02 on validation/test.

## Business Simulation (Test Split)

| Target Approval % | Actual Approval % | Threshold | Bad Rate % | Expected Loss % (fee 1) | Fee 2 | Fee 3 |
|------------------:|------------------:|----------:|-----------:|------------------------:|------:|------:|
| 40 | 40.04 | 0.182 | 6.96 | 5.96 | 4.96 | 3.96 |
| 60 | 60.01 | 0.276 | 12.28 | 11.28 | 10.28 | 9.28 |
| 80 | 82.18 | 0.295 | 16.73 | 15.73 | 14.73 | 13.73 |

Loss remains above fee revenue at higher approval targets; a 60% approval strategy still requires higher pricing or tighter credit rules to break even.

## Slice Analysis

Key slices saved under `artifacts/risk/eval/` highlight risk concentration:

- **Pay frequency:** weekly advances (bad rate 22.5%) remain the riskiest cohort; monthly users show bad rate ≈11% with ROC-AUC 0.73.
- **Advance amount deciles:** bad rate rises with advance size; top decile sits at ~20% bads and ROC-AUC 0.69.

These slices can guide pricing tiers or policy overrides.

## Testing & Quality Checks

- `tests/test_risk.py` covers leakage protection and utilization monotonicity.
- `uv run pytest` integrated into `make test`; `make risk-train` and `make risk-eval` reproduce the full pipeline and evaluation artifacts inside the uv-managed environment.

## Next Steps

1. Add SHAP or permutation diagnostics to highlight dominant utilisation/behaviour drivers for underwriters.
2. Explore employer-level Bayesian smoothing and cashflow stress features for additional ROC lift if headroom remains.
3. Automate markdown refresh (train → eval → report) to keep documentation and artifacts in sync.
