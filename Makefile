# llm-data-leak-guard — developer tasks
# Usage: make install && make test && make bench && make run

PYTHON      ?= python3
VENV        ?= .venv
BIN         := $(VENV)/bin
HOST        ?= 127.0.0.1
PORT        ?= 8000

.DEFAULT_GOAL := help

.PHONY: help install test bench run clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create venv, install deps and the spaCy English model
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/python -m spacy download en_core_web_sm

test: ## Run the test suite (one test per leak class)
	$(BIN)/python -m pytest -q

bench: ## Measure real redaction latency over the synthetic corpus
	$(BIN)/python -m bench.run

run: ## Start the FastAPI guard proxy locally
	$(BIN)/python -m uvicorn proxy.app:app --host $(HOST) --port $(PORT) --reload

clean: ## Remove caches and the virtualenv
	rm -rf $(VENV) .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
