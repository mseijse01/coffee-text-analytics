# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Virtual environment is at `~/.virtualenvs/coffee-analytics/`. The Makefile uses this path directly — use `make` targets instead of calling `python` directly when possible, or activate with `source ~/.virtualenvs/coffee-analytics/bin/activate`.

## Commands

### Running the Pipeline
```bash
python main.py --steps all                          # Full pipeline
python main.py --steps preprocess                   # Preprocessing only
python main.py --steps features                     # Feature extraction only
python main.py --steps select                       # Feature selection only
python main.py --steps train                        # Model training only
python main.py --steps train --models xgboost ridge # Specific models
python main.py --steps visualize                    # Visualization only
```

### Testing
```bash
make test                                              # Safe lightweight tests (default)
make test-one FILE=tests/test_exceptions.py            # Single test file
make test-fast                                         # Skip slow + heavy_ml markers
make test-full                                         # All 16 test files with RAM monitoring
pytest tests/ -p no:cacheprovider -c pytest-fast.ini  # Fast config (no coverage, quieter)
pytest tests/ --cov=src --cov-report=term-missing      # With coverage
```

### Linting & Formatting
```bash
make format          # Auto-fix: black + isort (USE THIS, not --check versions)
make lint            # lint-syntax + lint-style checks
make ci-test         # lint + test (simulates CI pipeline locally)
mypy src/data/loader.py src/data/preprocessing.py src/features/feature_manager.py src/models/regressors.py src/models/evaluator.py  # Type checking (5 core files only)
```

### Dependency Management
```bash
# Development: Use flexible versions for active development
pip install -r requirements.txt

# Production/Reproducibility: Use locked versions for exact reproducibility
pip install -r requirements-lock.txt

# Pre-commit hooks (auto-fixes formatting on commit):
# Installed automatically via .pre-commit-config.yaml
# Runs: black, isort, mypy (5 core modules), trailing-whitespace, end-of-file-fixer, check-yaml
```

### Other Utilities
```bash
python validate_15_percent_methodology.py              # Thesis compliance validation (~4 min)
python validate_15_percent_methodology.py --sample_size=50  # 50% sample
make clean-cache                                       # Delete cache/ to force feature re-extraction
make clean-models                                      # Delete models/*.pkl
mlflow ui --port 5000                                  # View experiment runs
python -m config.cli --validate                        # Validate configuration
```

## Architecture

This is a **research ML pipeline** for analyzing consumer coffee reviews (CoffeeReview.com dataset, ~6,400 rows). The goal is predicting coffee quality ratings from text using NLP + regression models.

### Pipeline Flow
`data/raw/coffee_clean.csv` → preprocessing → feature extraction → feature selection → model training → MLflow logging → visualization/output

### Source Layout (`src/`)

- **`config/`** — Environment-aware configuration system (dev/prod/test/cicd). Use `python -m config.cli` to inspect.
- **`data/`** — Data loading and preprocessing. Uses **Polars** as the primary DataFrame library; Pandas is used only as a sklearn compatibility layer.
- **`features/`** — Modular feature extractors (TF-IDF, BERT/DistilBERT embeddings, LDA/NMF topic models, sentiment). `feature_manager.py` orchestrates all extractors; `feature_selector.py` (LASSO-based) reduces ~3,840 features down to ~279.
- **`models/`** — Six regression models: Linear, Ridge, LASSO, RandomForest, XGBoost, MNIR (Multinomial Inverse Regression). `evaluator.py` handles metrics and SHAP analysis.
- **`experiment/`** — MLflow + Optuna integration. MLflow uses a PostgreSQL backend + MinIO S3 storage.
- **`utils/`** — Caching system for expensive feature extraction, SHAP utilities, performance profiling.
- **`visualization/`** — Plotly/Matplotlib/Seaborn output for figures saved to `output/`.
- **`exceptions.py`** — Centralized exception hierarchy. All custom exceptions inherit from `CoffeeAnalyticsError`.

### Key Design Decisions
- **Type safety**: The 5 core files (`loader.py`, `preprocessing.py`, `feature_manager.py`, `regressors.py`, `evaluator.py`) are fully type-annotated and checked with mypy in CI. This is a gradual migration strategy—non-core modules are not yet annotated.
- **Caching**: Feature extraction (especially BERT) is expensive. The `cache/` directory stores extracted features as serialized files. Delete cache to force re-extraction (`make clean-cache`).
- **Polars-first**: Data processing uses Polars for performance; only converts to Pandas at sklearn boundaries.
- **Thesis compliance**: The 15% sample validation script exists specifically to verify the research methodology matches thesis requirements. Don't break this workflow.
- **Best model**: XGBoost achieves R²=0.9453. MNIR is included for research/interpretability (not performance).

### CI Pipeline Behavior
- **Main CI job** (`test`): only runs `tests/test_data_processing.py` (fast, keeps CI under 10 min). Coverage threshold: 15%.
- **Integration tests**: run only on push to `main`, not on PRs.
- **mypy**: checks only the 5 core files listed above.

### Test Markers
Tests use pytest markers to separate concerns:
- `slow` / `heavy_ml` — Skip these for fast iteration (`-m "not slow and not heavy_ml"`)
- `integration` — Require full pipeline components loaded
- `contract` — API contract tests for feature selector
- `unit` / `edge_case` / `error_handling` — Standard unit test classifications
- `mlflow` — Require MLflow server
- `methodology` / `performance` — Research validation tests
