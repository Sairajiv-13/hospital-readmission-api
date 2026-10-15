PYTHON ?= python3
VENV := .venv
ARTIFACTS ?= artifacts
DATA ?= data/diabetic_data.csv

.PHONY: setup train test lint serve docker-build

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

train:
	$(VENV)/bin/python -m readmission.train --data $(DATA) --artifacts $(ARTIFACTS)

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check src tests

serve:
	$(VENV)/bin/uvicorn readmission.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t hospital-readmission-api .
