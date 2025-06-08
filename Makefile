# ML Training & Serving Pipeline - Essential Commands
.PHONY: help install train-local serve-local test clean format lint

help: ## Show this help message
	@echo "ML Training & Serving Pipeline - Commands"
	@echo "========================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
install: ## Install dependencies
	pip install -r requirements.txt

setup-conda: ## Setup conda environment
	conda create -n ml-pipeline python=3.11 -y
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && pip install -r requirements.txt

##@ Training
train-local: ## Train DLRM model locally (quick test)
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python training/train_dlrm.py --local --samples 10000 --epochs 5

train-local-full: ## Train DLRM model locally (full dataset)
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python training/train_dlrm.py --local --samples 200000 --epochs 20

##@ Azure Training (Optional - for scaling up)
install-azure: ## Install Azure ML dependencies
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && pip install azure-ai-ml azure-identity

train-azure-small: ## Train on Azure ML (small dataset)
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python training/train_dlrm.py --azure --samples 100000 --epochs 10

train-azure-large: ## Train on Azure ML (large dataset)
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python training/train_dlrm.py --azure --samples 1000000 --epochs 30

test-azure: ## Test Azure ML connection
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python training/azure_integration/submit_job.py

##@ Serving (TODO: Implement after training works)
serve-local: ## Start local BentoML server
	@echo "🚧 Serving pipeline coming next..."
	# eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python serving/bentoml_service.py

##@ Development
test: ## Run basic tests
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python -m pytest tests/ -v

format: ## Format code
	black training/ serving/ shared/ examples/ tests/
	
lint: ## Lint code
	ruff check training/ serving/ shared/ examples/ tests/

clean: ## Clean generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf models/ *.log

##@ Demo
demo: ## Run end-to-end demo
	eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python examples/end_to_end_demo.py

##@ Info
check-env: ## Check conda environment
	@eval "$(conda shell.bash hook)" && conda activate ml-pipeline && python -c "import torch, pandas, numpy; print('✅ Dependencies OK')"

show-models: ## Show trained models
	@ls -la models/ 2>/dev/null || echo "No models directory found"