# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Virtual environment is at `~/.virtualenvs/coffee-analytics/`. The Makefile uses this path directly — use `make` targets instead of calling `python` directly when possible, or activate with `source ~/.virtualenvs/coffee-analytics/bin/activate`.

## Commands

### Running the Pipeline
```bash
python main.py --steps all                           # Full pipeline
python main.py --steps all --sample_fraction 0.15   # Full pipeline on 15% sample (RAM-friendly)
python main.py --steps all --sample_size 400        # Full pipeline on exact row count
python main.py --steps preprocess                   # Preprocessing only
python main.py --steps features                     # Feature extraction only
python main.py --steps select                       # Feature selection only
python main.py --steps train                        # Model training only
python main.py --steps train --models xgboost ridge # Specific models
python main.py --steps visualize                    # Visualization only
```

### Testing

**Quick Test Suite (Recommended)** — 35 focused unit tests, ~5 seconds:
```bash
bash run_tests_externally.sh quick   # Recommended: 35 tests, all passing
```
Tests: MNIR core (18) + focused (11) + cache (6). No flaky dependencies, deterministic.

**Full Test Suite**:
```bash
bash run_tests_externally.sh full    # All ~400 tests across 16 test files (~30 min, heavy RAM)
make test                            # Safe lightweight tests (default)
make test-one FILE=tests/test_exceptions.py   # Single test file
make test-fast                       # Skip slow + heavy_ml markers
make test-full                       # All test files with RAM monitoring
pytest tests/ -p no:cacheprovider    # Fast config (no coverage, quieter)
pytest tests/ --cov=src --cov-report=term-missing  # With coverage
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

### Examples & Demos

**Optuna Hyperparameter Optimization Demos** (`examples/`):

```bash
# Quick demo (~5-10 min, 25 trials, good for testing)
python examples/research_optuna_quick_demo.py

# Full demo (~20-30 min, 50+ trials, production-grade optimization)
python examples/research_optuna_demo.py
```

**When to use each:**
- `quick_demo.py` — Test changes, verify pipeline works, quick iteration
- `demo.py` — Full research-grade optimization, portfolio showcase, publication quality

### Utilities & Maintenance Scripts

**Available in `scripts/`:**

```bash
# Clean output directories before fresh pipeline run
python scripts/clean_outputs.py --dry-run     # Preview what will be deleted
python scripts/clean_outputs.py --confirm     # Clean without prompting

# Generate/update API documentation
python scripts/generate_docs.py
```

**Historical Run Data:**
- Keep `mlruns/` locally for experiment benchmarking (gitignored, not in repo)
- Keep past model artifacts in `models/` — useful for comparing old vs. new performance
- Use `mlflow ui` to browse and compare historical runs

### Other Utilities
```bash
python validate_15_percent_methodology.py              # Thesis compliance validation (~4 min)
python validate_15_percent_methodology.py --sample_size=50  # 50% sample
python validate_15_percent_and_save.py                 # Validate on 15% sample + save model artifacts
python validate_30_percent_and_save.py                 # Validate on 30% sample + save model artifacts
make validate-quick                                    # Quick validation on 5% sample (~30 sec)
make train-xgboost                                     # Train XGBoost only (best model)
make clean-cache                                       # Delete cache/ to force feature re-extraction
make clean-models                                      # Delete models/*.pkl
mlflow ui --port 5000                                  # View experiment runs
python -m config.cli --validate                        # Validate configuration
COFFEE_ENV=production python main.py --steps all       # Switch environment (dev/prod/test/cicd)
```

### Serving Layer
```bash
make serve             # Start FastAPI on http://localhost:8000 (requires make train first)
make serve-docker      # Build Dockerfile.serving and run container on port 8000
```

The serving layer lives in `src/serving/`: `app.py` (FastAPI routes + lifespan), `predictor.py` (loads TF-IDF/LASSO/XGBoost artifacts), `schemas.py` (Pydantic request/response models). Run from `src/` root via `uvicorn serving.app:app`.

### MLflow Infrastructure (PostgreSQL + MinIO)
```bash
docker-compose -f mlflow_setup/docker-compose.yml up   # Start MLflow server + artifact storage
# MLflow UI: http://localhost:5555, MinIO console: http://localhost:9001
```

## Architecture

This is a **research ML pipeline** for analyzing consumer coffee reviews (CoffeeReview.com dataset, ~6,400 rows). The goal is predicting coffee quality ratings from text using NLP + regression models.

### Data Schema
- **Source**: `data/raw/coffee_clean.csv` — ~6,400 rows, filtered to ~2,440 after minimum rating cutoff
- **Target**: `rating` (80–100 scale)
- **Text inputs**: `desc_1`, `desc_2`, `desc_3` (three review description columns)
- **Sensory attributes**: `aroma`, `acid`, `body`, `flavor`, `aftertaste` (kept separate per thesis methodology — see `pipeline/constants.py:EXCLUDE_COLUMNS`)
- **Train/test split**: 70/30 stratified by rating bins

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

⚠️ **LASSO Feature Selection Architecture Issue** (2026-04-21, documented)
- **Status**: Known issue, analysis complete (see `docs/ARCHITECTURE_ISSUE_LASSO_FEATURE_SELECTION.md`)
- **Problem**: Main pipeline correctly handles sensory column exclusion; validation scripts diverge from thesis methodology
- **Impact**: Validation scripts pass all features to models (including sensory), not suitable for thesis comparison
- **Recommendation**: Use validation scripts for serving-layer artifacts only; use main pipeline with `--sample_fraction 0.15` for thesis-compliant runs

### Pipeline Modes: Main vs. Validation Scripts

There are two paths to generate model artifacts — they serve different purposes and produce different evaluation semantics:

**Main pipeline** (`python main.py --steps all [--sample_fraction 0.15]`):
- Follows thesis methodology: sensory + raw categorical columns are excluded from the LASSO input (via `EXCLUDE_COLUMNS` in `pipeline/constants.py`), then re-joined after selection
- Each feature group (flavor, text, categorical) can be evaluated independently
- Saves all artifacts (model pickles + LASSO selector + TF-IDF vectorizer) to `models/`
- Use `--sample_fraction 0.15` for a RAM-friendly end-to-end run outside Claude Code

**Validation scripts** (`validate_15_percent_and_save.py`, `validate_30_percent_and_save.py`):
- Purpose: fast convenience path for generating serving-layer artifacts in one command
- Pass combined features (text + sensory + categorical) to all models simultaneously
- Linear R²≈1.0 is **expected** here — sensory features alone give R²=0.998 (thesis table); these scripts do not reproduce the per-feature-group results shown in the thesis
- MNIR diverges from thesis design (thesis: text features → sensory attributes; scripts: full combined matrix)
- Use these when you need artifacts quickly and don't need thesis-comparable metrics

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

## Next Phase: Integration Bridge

The FastAPI serving layer is **implemented** (`src/serving/`). The next priority is integrating it with the sibling project:

### Roadmap
1. ~~**FastAPI Serving Layer**~~ — done (`make serve`)
2. **PostgreSQL Migration** (`coffee-database` sibling at `/Users/seijas/Code/coffee-database`) — persist scraped beans in a real database
3. **Integration Bridge** — have the scraper call `/predict` to auto-rate new beans

See `docs/NEXT_STEPS_BRAINSTORM.md` for full analysis.
