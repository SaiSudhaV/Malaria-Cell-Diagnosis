.PHONY: help setup install test lint clean docker-build docker-run prefect-start mlflow-start api-start all-services

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Setup project environment
	python scripts/setup_environment.py

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run tests
	pytest tests/ -v --cov=src --cov=api

lint: ## Run linting
	pylint src/ api/ scripts/ --disable=all --enable=E,F

clean: ## Clean cache and temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache htmlcov .coverage

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t malaria-diagnosis:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 malaria-diagnosis:latest

prefect-start: ## Start Prefect server
	prefect server start

mlflow-start: ## Start MLflow server
	mlflow ui --host 0.0.0.0 --port 5000

api-start: ## Start FastAPI server
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

all-services: ## Start all services with Docker Compose
	docker-compose up -d

preprocess: ## Run data preprocessing
	python scripts/preprocess_data.py

evaluate: ## Run model evaluation
	python scripts/evaluate.py

register-model: ## Register model in MLflow
	python scripts/register_model_mlflow.py

run-workflow: ## Run Prefect workflow (usage: make run-workflow WORKFLOW=preprocessing)
	python scripts/run_prefect_workflows.py $(WORKFLOW)

dvc-repro: ## Reproduce DVC pipeline
	dvc repro

ci-local: ## Run CI pipeline locally
	make lint
	make test
	make docker-build