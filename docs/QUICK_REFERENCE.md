# Coffee Text Analytics - Quick Reference

**Last Updated**: 2026-04-20
**Project Status**: ✅ **Portfolio-Ready** — XGBoost R²=0.9453, 100% thesis compliance

---

## Model Performance Summary

| Model | R² | RMSE | MAE | Status |
|-------|-----|------|-----|--------|
| **XGBoost** | **0.9453** | **0.4103** | **0.2152** | ⭐ Best |
| Ridge | 0.9259 | 0.4775 | 0.3801 | Excellent |
| LASSO | 0.8897 | 0.5825 | 0.4623 | Strong |
| Random Forest | 0.8675 | 0.6386 | 0.3590 | Good |
| Linear | 0.8173 | 0.7497 | 0.6101 | Baseline |

**MNIR (Interpretability Analysis):**
- Acidity: R² = 0.9389
- Aftertaste: R² = 0.8420
- Body: R² = 0.7966
- Aroma: R² = 0.7834
- Flavor: R² = 0.5789

---

## Quick Start

### Validate Thesis Methodology (4 minutes)
```bash
python validate_15_percent_methodology.py
```

### Run Full Pipeline
```bash
python main.py --steps all
```

### Run Specific Steps
```bash
python main.py --steps preprocess      # Preprocessing only
python main.py --steps features        # Feature extraction only
python main.py --steps select          # Feature selection only
python main.py --steps train           # Model training only
python main.py --steps train --models xgboost ridge  # Specific models
python main.py --steps visualize       # Visualization only
```

---

## Research Methodology

**Data Source**: CoffeeReview.com, ~6,400 reviews (2,440 used after filtering)

**Target Variable**: Coffee rating (80–100 scale)

**Train/Test Split**: 70/30 stratified by rating bins

### Feature Engineering Pipeline

**Raw Features**: ~3,840 total

**Selected Features**: 279 (92.7% dimensionality reduction via LASSO)

**Feature Types**:
1. **TF-IDF Vectorization** (600 features) — Unigrams, bigrams, trigrams
2. **BERT Embeddings** (2,304 features) — DistilBERT 768-dim semantic representations
3. **GloVe Embeddings** (900 features) — Pre-trained 300-dim word vectors
4. **Topic Modeling** (30 features) — LDA + NMF thematic analysis
5. **Sentiment Analysis** (6 features) — DistilBERT sentiment scores
6. **Categorical & Numeric** (15+ features) — Origin, roast level, sensory attributes

### Key Findings

**Text features dominate**: BERT embeddings and TF-IDF features are most predictive of coffee ratings, followed by sentiment scores and topic features.

**Topic analysis reveals**: Distinct themes including origin characteristics, processing methods, flavor profiles, and brewing recommendations.

**Sentiment-rating correlation**: Strong relationship between sentiment and ratings:
- Positive sentiment → higher ratings (8.5+)
- Negative sentiment → lower ratings (<7.0)

---

## Architecture Overview

### Pipeline Flow
```
data/raw/coffee_clean.csv
  → preprocessing
  → feature extraction
  → feature selection (LASSO)
  → model training
  → MLflow logging
  → visualization/output
```

### Core Components

**Data Processing** (`src/data/`)
- Polars-first approach with Pandas compatibility layer for sklearn
- Dual-mode loading: local files or MinIO S3-compatible storage

**Feature Extraction** (`src/features/`)
- TF-IDF, BERT, GloVe, LDA, NMF, sentiment extractors
- Caching system for expensive operations (especially BERT)

**Model Training** (`src/models/`)
- 6 regression models: Linear, Ridge, LASSO, Random Forest, XGBoost, MNIR
- Optuna-based hyperparameter optimization (TPE algorithm)
- SHAP analysis for feature importance

**Experiment Tracking** (`src/experiment/`)
- MLflow for tracking runs, parameters, metrics
- PostgreSQL backend + MinIO S3 artifact storage

---

## Development Setup

### Prerequisites
- Python 3.9+
- 8GB+ RAM (for BERT embeddings)
- GPU optional (CUDA for faster processing)

### Virtual Environment
```bash
# Located at ~/.virtualenvs/coffee-analytics/
source ~/.virtualenvs/coffee-analytics/bin/activate
```

### Install Dependencies
```bash
# Development (flexible versions)
pip install -r requirements.txt

# Production (pinned versions for reproducibility)
pip install -r requirements-lock.txt
```

### Verify Installation
```bash
make test  # Run lightweight test suite
```

---

## Testing & Quality

### Test Suites
- **Fast tests**: `make test-safe` (default, ~30 sec)
- **Full tests**: `make test-full` (all 16 files, ~2 min)
- **Specific file**: `make test-one FILE=tests/test_data_processing.py`

### Code Quality
- **Linting**: `make lint` (flake8 checks)
- **Formatting**: `make format` (black + isort auto-fix)
- **Type checking**: `mypy` on 5 core modules (gradual adoption)
- **Pre-commit hooks**: Auto-fixes formatting before commit

### Coverage
- Target: 18%+ (fast tests)
- Full pipeline coverage: ~96.7%

---

## Infrastructure

### Local Development
- **Cache**: `cache/` — Feature extraction caching (auto-generated)
- **Models**: `models/` — Trained model artifacts (.pkl files)
- **Output**: `output/` — Visualizations and results

### Docker & MLflow
- **Training container**: `Dockerfile.training` — Containerized ML pipeline
- **MLflow server**: `http://localhost:5555` (local development)
- **MinIO storage**: `http://localhost:9001` (S3-compatible artifacts)

### Deployment
```bash
# Docker Compose (local development)
docker-compose -f mlflow_setup/docker-compose.yml up

# Build training container
docker build -f Dockerfile.training -t coffee-analytics:latest .
```

---

## Troubleshooting

### Clear Feature Cache
```bash
make clean-cache
```
Forces re-extraction of all features (useful if feature code changes)

### Clear Models
```bash
make clean-models
```
Deletes trained model artifacts

### View MLflow Experiments
```bash
mlflow ui --port 5000
```

### Validate Configuration
```bash
python -m config.cli --validate
```

---

## Documentation

- **README.md** — Project overview and setup
- **MODEL_CARD.md** — Model documentation for hiring managers
- **docs/FEATURE_SELECTION_GUIDE.md** — Technical deep-dive on LASSO feature selection
- **docs/thesis.md** — Original thesis submission (research artifact)
- **CLAUDE.md** — Developer environment setup
- **mlflow_setup/README.md** — MLflow and Docker setup details

---

## Key References

**Thesis**: "Leveraging Text Analytics and Predictive Modeling to Analyze Consumer Coffee Reviews: A Data-Driven Approach"
**Author**: Marcelo Seijas
**Institution**: Erasmus University Rotterdam, Erasmus School of Economics
**Program**: Data Science and Marketing Analytics
**Year**: 2024

**Supervisor**: Eoghan O'Neill
**Second Assessor**: Sean Brüggemann
