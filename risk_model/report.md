# Repayment Risk Model Report

## Overview

- **Objective:** Estimate `p_bad` for each advance at request time using the synthetic 18‑month Earned Wage Access dataset.
- **Final model:** XGBoost classifier with isotonic calibration wrapped in a preprocessing pipeline (`artifacts/risk/model.joblib`).
- **Test performance:** ROC-AUC 0.683, PR-AUC 0.308, Brier 0.149, ECE 0.027 on the held-out 3-month window.

## Data & Leakage Safeguards

- Dataset generated via `generate_ewa_synth.py` (seed 42) and ingested with schema validation (`risk_model/datasets.py`).
- Time-aware splits: earliest 12 months for training (18,468 advances), next 3 for validation (4,599), final 3 for test (4,451). Label distribution remains ~19% bad across splits.
- Feature windows use only transactions strictly before `requested_at`; unit tests (`tests/test_risk.py::test_cashflow_features_exclude_post_request_transactions`) enforce this leakage guard.
- Monthly cohort monitoring (`artifacts/risk/eval/cohort_stats.csv`) confirms stable bad-rate band 18–21% with slight lift in the final months.

## Feature Summary

Feature engineering (`risk_model/features.py`) assembles:

- **Income cadence:** days since last payroll, median cycle length, last & trailing-3 payroll amounts, pay-frequency dummies. Weekly users show the highest bad rate (22.8%).
- **Cashflow health:** rolling net cash over 14/30/60/90 days, 30-day credit/debit volume, transaction counts, and daily volatility.
- **Utilisation & indebtedness:** advance amount ratios to payroll and salary, active advance counts/recency.
- **Behavioural history:** rolling bad/late rates, prior advances completed, days since last advance.

These features populate 22 numeric + 1 categorical columns in the pipeline metadata.

## Model Training & Selection

Candidate models were trained on the preprocessed training split and compared on validation metrics:

| Model                   | ROC-AUC | PR-AUC | Brier |
|-------------------------|--------:|-------:|------:|
| Logistic Regression      | 0.649   | 0.267  | 0.145 |
| HistGradientBoosting     | 0.657   | 0.265  | 0.144 |
| **XGBoost (selected)**   | **0.660** | **0.271** | 0.144 |

The XGBoost variant offered the best validation ROC-AUC and PR-AUC, clearing the 0.68 target on the subsequent test set.

## Evaluation Metrics

| Split | ROC-AUC | PR-AUC | Brier | Log Loss | ECE |
|-------|--------:|-------:|------:|---------:|----:|
| Train | 0.827 | 0.607 | 0.136 | 0.429 | 0.082 |
| Val   | 0.665 | 0.276 | 0.143 | 0.453 | 0.017 |
| **Test**  | **0.683** | **0.308** | **0.149** | **0.463** | **0.027** |

Calibration curve analysis (see `artifacts/risk/eval/plots/calibration_test.png`) shows mild under-confidence in the lowest probability bins and slight over-confidence above 0.24. Post-calibration ECE stays below 0.03 on validation/test, though the training curve remains tighter (ECE 0.082) due to isotonic smoothing.

## Business Simulation (Test Split)

| Target Approval % | Actual Approval % | Threshold | Bad Rate % | Expected Loss % (fee 1) | Fee 2 | Fee 3 |
|------------------:|------------------:|----------:|-----------:|------------------------:|------:|------:|
| 40 | 40.04 | 0.181 | 8.08 | 7.08 | 6.08 | 5.08 |
| 60 | 60.03 | 0.239 | 12.65 | 11.65 | 10.65 | 9.65 |
| 80 | 80.12 | 0.271 | 16.97 | 15.97 | 14.97 | 13.97 |

Loss remains above fee revenue at higher approval targets; a 60% approval strategy requires either higher pricing or tighter underwriting controls to break even.

## Slice Analysis

Key slices saved under `artifacts/risk/eval/` highlight risk concentration:

- **Pay frequency:** weekly advances (bad rate 22.8%) exhibit lower ROC-AUC (0.647) and higher loss; monthly users remain the safest (bad rate 10.9%, Brier 0.091).
- **Advance amount deciles:** bad rate rises modestly with larger amounts; decile 10 peaks at 20.4% with ROC-AUC 0.679.

These slices can guide pricing tiers or policy overrides.

## Testing & Quality Checks

- `tests/test_risk.py` covers leakage protection and utilization monotonicity.
- `uv run pytest` integrated into `make test`; `make risk-train` and `make risk-eval` reproduce the full pipeline and evaluation artifacts inside the uv-managed environment.

## Next Steps

1. Enrich behavioural features with smoothed lookbacks (e.g., exponential decay) and assess incremental lift.
2. Add SHAP-based diagnostics or permutation importance to surface actionable drivers for underwriting.
3. Integrate business thresholds into automated reporting (e.g., Markdown fragments) to keep this document reproducible.
