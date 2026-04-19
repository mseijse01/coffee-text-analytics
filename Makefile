.PHONY: help test test-safe test-full lint format clean train serve validate
.DEFAULT_GOAL := help

# Configuration
PYTHON := ~/.virtualenvs/coffee-analytics/bin/python
VENV := ~/.virtualenvs/coffee-analytics

# Colors for output
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
NC := \033[0m

help: ## Show this help message
	@echo "$(BOLD)☕ Coffee Text Analytics - Makefile$(NC)"
	@echo ""
	@echo "$(BOLD)Testing:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(test|lint|format)"
	@echo ""
	@echo "$(BOLD)Development:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(train|serve|validate|clean|venv)"
	@echo ""

# ============================================================================
# TESTING TARGETS
# ============================================================================

test: test-safe ## Run safe tests (lightweight, no heavy ML, no RAM crashes)

test-safe: ## Run lightweight unit tests (safest option for M1 Pro)
	@echo "$(BOLD)🧪 Running safe tests (lightweight)...$(NC)"
	@$(PYTHON) run_tests.py

test-full: ## Run full test suite (all 16 test files, file-by-file with RAM monitoring)
	@echo "$(BOLD)🧪 Running full test suite...$(NC)"
	@$(PYTHON) run_tests.py --batch

test-fast: ## Run fast tests only (skip slow and heavy_ml markers)
	@echo "$(BOLD)⚡ Running fast tests only...$(NC)"
	@$(PYTHON) -m pytest tests/ -m "not slow and not heavy_ml" -q --tb=short

test-one: ## Run a single test file (usage: make test-one FILE=tests/test_exceptions.py)
	@echo "$(BOLD)🧪 Running single test file...$(NC)"
	@$(PYTHON) -m pytest $(FILE) -v --tb=short

test-coverage: ## Generate HTML coverage report
	@echo "$(BOLD)📊 Generating coverage report...$(NC)"
	@$(PYTHON) -m pytest tests/test_data_processing.py tests/test_exceptions.py \
		--cov=src --cov-report=html --cov-report=term-missing -q
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/index.html$(NC)"

# ============================================================================
# LINTING & FORMATTING TARGETS
# ============================================================================

lint: lint-syntax lint-style ## Run all linting checks

lint-syntax: ## Check for syntax errors (strict)
	@echo "$(BOLD)🔍 Checking syntax errors...$(NC)"
	@$(PYTHON) -m flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

lint-style: ## Check style and complexity warnings
	@echo "$(BOLD)🔍 Checking style and complexity...$(NC)"
	@$(PYTHON) -m flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

lint-imports: ## Check import sorting
	@echo "$(BOLD)🔍 Checking import sorting...$(NC)"
	@$(PYTHON) -m isort --check-only src/ tests/

lint-format: ## Check code formatting
	@echo "$(BOLD)🔍 Checking code formatting...$(NC)"
	@$(PYTHON) -m black --check src/ tests/

format: ## Auto-format code (black + isort)
	@echo "$(BOLD)✨ Formatting code...$(NC)"
	@$(PYTHON) -m black src/ tests/
	@$(PYTHON) -m isort src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

# ============================================================================
# TRAINING & PIPELINE TARGETS
# ============================================================================

train: ## Run full pipeline (preprocess → features → select → train → visualize)
	@echo "$(BOLD)🚀 Running full training pipeline...$(NC)"
	@$(PYTHON) main.py --steps all

train-preprocess: ## Preprocessing only
	@echo "$(BOLD)🚀 Running preprocessing...$(NC)"
	@$(PYTHON) main.py --steps preprocess

train-features: ## Feature extraction only
	@echo "$(BOLD)🚀 Running feature extraction...$(NC)"
	@$(PYTHON) main.py --steps features

train-select: ## Feature selection only
	@echo "$(BOLD)🚀 Running feature selection...$(NC)"
	@$(PYTHON) main.py --steps select

train-models: ## Model training only
	@echo "$(BOLD)🚀 Running model training...$(NC)"
	@$(PYTHON) main.py --steps train

train-xgboost: ## Train XGBoost only (best model, R²=0.9453)
	@echo "$(BOLD)🚀 Running XGBoost training...$(NC)"
	@$(PYTHON) main.py --steps train --models xgboost

train-visualize: ## Visualization only
	@echo "$(BOLD)🚀 Running visualization...$(NC)"
	@$(PYTHON) main.py --steps visualize

# ============================================================================
# VALIDATION & ANALYSIS TARGETS
# ============================================================================

validate: ## Validate thesis methodology compliance (15% sample, ~4 min)
	@echo "$(BOLD)📋 Validating thesis methodology...$(NC)"
	@$(PYTHON) validate_15_percent_methodology.py

validate-quick: ## Quick validation on 5% sample (~30 sec)
	@echo "$(BOLD)⚡ Quick methodology validation (5% sample)...$(NC)"
	@$(PYTHON) validate_15_percent_methodology.py --sample_size=5

validate-config: ## Validate configuration
	@echo "$(BOLD)🔧 Validating configuration...$(NC)"
	@$(PYTHON) -m config.cli --validate

# ============================================================================
# DEVELOPMENT & UTILITY TARGETS
# ============================================================================

clean: ## Clean output directories and caches
	@echo "$(BOLD)🧹 Cleaning outputs...$(NC)"
	@$(PYTHON) clean_outputs.py --dry-run
	@read -p "Proceed with cleanup? (y/n) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON) clean_outputs.py --confirm; \
	fi

clean-cache: ## Clear feature extraction cache (forces recomputation)
	@echo "$(BOLD)🧹 Clearing feature cache...$(NC)"
	@rm -rf cache/
	@echo "$(GREEN)✓ Cache cleared$(NC)"

clean-models: ## Clear trained models
	@echo "$(BOLD)🧹 Clearing trained models...$(NC)"
	@rm -f models/*.pkl
	@echo "$(GREEN)✓ Models cleared$(NC)"

clean-all: clean clean-cache clean-models ## Clean everything

mlflow: ## Start MLflow UI (port 5000)
	@echo "$(BOLD)📊 Starting MLflow UI...$(NC)"
	@$(PYTHON) -m mlflow ui --port 5000
	@echo "$(GREEN)✓ MLflow running at http://localhost:5000$(NC)"

serve: ## Start FastAPI serving layer (placeholder for future)
	@echo "$(BOLD)🚀 Starting API server...$(NC)"
	@echo "$(YELLOW)Note: FastAPI serving layer not yet implemented$(NC)"
	@echo "Coming in Task 4: Add FastAPI serving endpoint"

# ============================================================================
# ENVIRONMENT TARGETS
# ============================================================================

venv-show: ## Show current venv information
	@echo "$(BOLD)🐍 Virtual Environment:$(NC)"
	@echo "  Path: $(VENV)"
	@echo "  Python: $$($(PYTHON) --version)"
	@$(PYTHON) -c "import sys; print('  Packages: ' + str(len(set([p.split('==')[0] for p in open('requirements.txt').read().split() if p and not p.startswith('#')]))))"

venv-activate: ## Show how to activate venv
	@echo "$(BOLD)🐍 To activate virtual environment:$(NC)"
	@echo "  source $(VENV)/bin/activate"

# ============================================================================
# CI/LOCAL TESTING PIPELINE
# ============================================================================

ci-test: lint test ## Run linting + tests (CI pipeline simulation)
	@echo "$(BOLD)✓ CI pipeline complete$(NC)"

ci-full: lint test-fast validate-config ## Full CI with validation
	@echo "$(BOLD)✓ Full CI pipeline complete$(NC)"

# ============================================================================
# CONVENIENCE SHORTCUTS
# ============================================================================

watch-tests: ## Watch for changes and re-run tests (requires watchmedo)
	@echo "$(BOLD)👀 Watching for changes...$(NC)"
	@watchmedo shell-command --patterns="*.py" --recursive --command="make test-safe" .

all: lint test validate ## Run lint + test + validate (full quality check)
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(NC)"
