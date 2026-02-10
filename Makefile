# ML Training & Serving Pipeline - Essential Commands
.PHONY: help install install-dev train-local train-local-full install-azure train-azure-small train-azure-large test-azure serve-local serve test test-cov format lint type-check clean demo check-env show-models

help: ## Show this help message
	@echo "ML Training & Serving Pipeline - Commands"
	@echo "========================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

##@ Training
train-local: ## Train DLRM model locally (quick test)
	python training/train_dlrm.py --local --samples 10000 --epochs 5

train-local-full: ## Train DLRM model locally (full dataset)
	python training/train_dlrm.py --local --samples 200000 --epochs 20

##@ Azure Training (Optional - for scaling up)
install-azure: ## Install Azure ML dependencies
	pip install -e ".[azure]"

train-azure-small: ## Train on Azure ML (small dataset)
	python training/train_dlrm.py --azure --samples 100000 --epochs 10

train-azure-large: ## Train on Azure ML (large dataset)
	python training/train_dlrm.py --azure --samples 1000000 --epochs 30

test-azure: ## Test Azure ML connection
	python training/azure_integration/submit_job.py

##@ Serving
serve-local: ## Start local model serving
	python -m serving.model_loader

serve: ## Start FastAPI serving app
	uvicorn serving.app:app --reload --port 8000

##@ Development
test: ## Run tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --cov=training --cov=serving --cov=shared

format: ## Format code
	black training/ serving/ shared/ examples/ tests/

lint: ## Lint code
	ruff check training/ serving/ shared/ examples/ tests/

type-check: ## Run type checking
	mypy training/ serving/ shared/

clean: ## Clean generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf models/ *.log

##@ Demo
demo: ## Run end-to-end demo
	python examples/end_to_end_demo.py

##@ Info
check-env: ## Check environment
	python -c "import torch, pandas, numpy; print('Dependencies OK')"

show-models: ## Show trained models
	@ls -la models/ 2>/dev/null || echo "No models directory found"
