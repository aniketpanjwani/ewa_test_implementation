.PHONY: data risk-train risk-eval paydate-eval bot-run bot-eval lint test

UV ?= uv
PYTHON := $(UV) run python

# Generate synthetic dataset using the provided script.
data:
	$(PYTHON) generate_ewa_synth.py --users 2000 --employers 120 --months 18 --start 2024-01-01 --seed 42 --out ./data

# Train the repayment risk model.
risk-train:
	$(PYTHON) risk_model/train.py --data ./data --out ./artifacts/risk

# Evaluate the repayment risk model.
risk-eval:
	$(PYTHON) risk_model/evaluate.py --data ./data --artifacts ./artifacts/risk

# Placeholder targets for future tasks.
paydate-eval:
	$(PYTHON) paydate_model/evaluate.py --data ./data --out ./artifacts/paydate

bot-run:
	$(PYTHON) support_bot/app.py "How do fees work?"

bot-eval:
	$(PYTHON) support_bot/evaluate.py --docs ./docs --eval ./docs/bot_eval.jsonl

lint:
	$(UV) run ruff check .

# Run all tests once stubs are implemented.
test:
	$(UV) run pytest
