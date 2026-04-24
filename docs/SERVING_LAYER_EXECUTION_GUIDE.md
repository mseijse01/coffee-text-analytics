# Execution Guide: FastAPI Serving Layer

**For**: Haiku executing in a fresh session on `coffee-text-analytics`
**Goal**: Add a FastAPI serving layer so the trained XGBoost model is accessible via HTTP endpoint
**Working directory**: `/Users/seijas/Code/coffee-text-analytics`

---

## Before you start: verify prerequisites

Run this check first. If any file is missing, run `make train` before proceeding.

```bash
ls models/tfidf_vectorizer.pkl models/lasso_feature_selector.pkl models/xgboost_model.pkl
```

If those three files exist, proceed. If not:
```bash
source ~/.virtualenvs/coffee-analytics/bin/activate
python main.py --steps features select train
```

This takes ~20–40 min (BERT extraction is slow). The topic models (`lda_model.pkl`, `nmf_model.pkl`, `topic_vectorizer.pkl`) already exist in `models/` and don't need regeneration.

---

## Step 1 — Add dependencies to `requirements.txt`

Append these three lines to the end of `requirements.txt`:
```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
httpx>=0.24.0
```

Then install:
```bash
source ~/.virtualenvs/coffee-analytics/bin/activate
pip install fastapi "uvicorn[standard]" httpx
```

---

## Step 2 — Create `src/serving/__init__.py`

Create an empty file at `src/serving/__init__.py`.

---

## Step 3 — Create `src/serving/schemas.py`

Create `src/serving/schemas.py` with this exact content:

```python
from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    desc_1: str = Field(..., description="Primary tasting notes")
    desc_2: str = Field("", description="Secondary review notes")
    desc_3: str = Field("", description="Bottom-line conclusion")
    aroma: Optional[float] = Field(None, ge=0, le=10)
    acid: Optional[float] = Field(None, ge=0, le=10)
    body: Optional[float] = Field(None, ge=0, le=10)
    flavor: Optional[float] = Field(None, ge=0, le=10)
    aftertaste: Optional[float] = Field(None, ge=0, le=10)


class PredictResponse(BaseModel):
    rating: float = Field(..., description="Predicted coffee rating (80–100 scale)")
    model_name: str = "xgboost"
    r2_score: float = 0.9453


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    r2_score: float
    n_features_selected: int
    feature_extractors: list
```

---

## Step 4 — Create `src/serving/predictor.py`

### What this file does

Loads all trained artifacts once at app startup, then runs inference on each request. The loading sequence matters:
1. `TfidfExtractor.load_extractor(models_dir)` — loads `models/tfidf_vectorizer.pkl`
2. `TopicExtractor.load_models(models_dir)` — loads `models/lda_model.pkl`, `nmf_model.pkl`, `topic_vectorizer.pkl`
3. `CoffeeFeatureManager` is instantiated and the pre-loaded extractors are injected into it (do NOT call `fm.load_extractors()` — it has a bug where it calls `load_vectorizer()` on TF-IDF but the method is named `load_extractor()`)
4. BERT and sentiment extractors are pre-trained, they are already initialized by `CoffeeFeatureManager.__init__` and need no artifact loading
5. `LassoFeatureSelector.load_selector(path)` — classmethod, returns loaded selector
6. `pickle.load` the XGBoost model — returns a `CoffeeXGBoost` instance with `.predict(X)` method

### sys.path requirement

Files in `src/serving/` are two levels deep. To import from `src/` (e.g. `from features.feature_manager import ...`), they must add `src/` to sys.path:
```python
_SRC = Path(__file__).parent.parent   # src/serving/../../ = src/
_ROOT = _SRC.parent                   # project root
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))
```

### Create the file

```python
"""
CoffeePredictor: loads all trained artifacts at app startup, runs inference per request.
"""
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import polars as pl

# Add src/ and project root to sys.path so sibling packages resolve
_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from features.feature_manager import CoffeeFeatureManager
from features.feature_selector import LassoFeatureSelector
from features.tfidf_extractor import TfidfExtractor
from features.topic_extractor import TopicExtractor
from pipeline.constants import EXCLUDE_COLUMNS

logger = logging.getLogger(__name__)

MODELS_DIR = _ROOT / "models"

# Columns that are never model features — strip them before selector.transform()
_EXCLUDE = set(EXCLUDE_COLUMNS) | {"desc_1", "desc_2", "desc_3"}


class CoffeePredictor:
    def __init__(self) -> None:
        self.fm: Optional[CoffeeFeatureManager] = None
        self.selector: Optional[LassoFeatureSelector] = None
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """
        Load all artifacts. Called once from FastAPI lifespan on startup.
        BERT loading takes ~1-2 minutes — this is expected.
        """
        logger.info("Loading serving artifacts from %s", MODELS_DIR)

        # 1. TF-IDF extractor
        tfidf = TfidfExtractor({"models_dir": str(MODELS_DIR)})
        tfidf.load_extractor(str(MODELS_DIR))  # method is load_extractor, NOT load_vectorizer
        logger.info("TF-IDF vectorizer loaded")

        # 2. Topic extractor (LDA + NMF)
        topics = TopicExtractor({"models_dir": str(MODELS_DIR)})
        topics.load_models(str(MODELS_DIR))
        logger.info("Topic models loaded")

        # 3. Feature manager — inject pre-loaded extractors directly
        self.fm = CoffeeFeatureManager({
            "extractors": {
                "tfidf": True,
                "bert": True,       # pre-trained DistilBERT — no pkl to load
                "topics": True,
                "sentiment": True,  # pre-trained DistilBERT sentiment — no pkl to load
                "glove": False,     # disabled: requires large gensim download at runtime
            }
        })
        self.fm.extractors["tfidf"] = tfidf
        self.fm.extractors["topics"] = topics
        # bert and sentiment extractors were already created by CoffeeFeatureManager.__init__
        self.fm.is_fitted = True
        logger.info("Feature manager ready (BERT/sentiment loaded as pre-trained)")

        # 4. LASSO feature selector
        selector_path = MODELS_DIR / "lasso_feature_selector.pkl"
        self.selector = LassoFeatureSelector.load_selector(selector_path)
        logger.info("LASSO selector loaded")

        # 5. Regression model
        model_path = MODELS_DIR / "xgboost_model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("XGBoost model loaded — predictor ready")

        self._loaded = True

    def predict(
        self,
        desc_1: str,
        desc_2: str = "",
        desc_3: str = "",
        aroma: Optional[float] = None,
        acid: Optional[float] = None,
        body: Optional[float] = None,
        flavor: Optional[float] = None,
        aftertaste: Optional[float] = None,
    ) -> float:
        if not self._loaded:
            raise RuntimeError("CoffeePredictor.load() must be called before predict()")

        # Build single-row Polars DataFrame matching training data schema
        row = {"desc_1": [desc_1], "desc_2": [desc_2 or ""], "desc_3": [desc_3 or ""]}
        df = pl.DataFrame(row)

        # Extract all features — same pipeline as training
        features_df = self.fm.extract_all_features(
            df, text_columns=["desc_1", "desc_2", "desc_3"]
        )

        # Drop non-feature columns, convert to pandas for sklearn
        feature_cols = [c for c in features_df.columns if c not in _EXCLUDE]
        X = features_df.select(feature_cols).to_pandas()

        # Apply LASSO selection (reduces to ~279 features)
        X_selected = self.selector.transform(X)

        # Predict (model.predict returns np.ndarray)
        return float(self.model.predict(X_selected)[0])

    @property
    def is_loaded(self) -> bool:
        return self._loaded
```

---

## Step 5 — Create `src/serving/app.py`

```python
"""FastAPI serving layer for coffee rating prediction."""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException

from .predictor import CoffeePredictor
from .schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictor = CoffeePredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts on startup. BERT takes ~1-2 min — start-period in Docker is set accordingly."""
    logger.info("Startup: loading ML artifacts (BERT load may take 1-2 min)...")
    _predictor.load()
    logger.info("Startup complete. API ready.")
    yield
    # No teardown needed


app = FastAPI(
    title="Coffee Analytics API",
    description="Predict coffee quality ratings (80–100) from review text",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_predictor.is_loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name="xgboost",
        r2_score=0.9453,
        n_features_selected=279,
        feature_extractors=["tfidf", "bert", "topics", "sentiment"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not _predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    try:
        rating = _predictor.predict(
            desc_1=request.desc_1,
            desc_2=request.desc_2,
            desc_3=request.desc_3,
            aroma=request.aroma,
            acid=request.acid,
            body=request.body,
            flavor=request.flavor,
            aftertaste=request.aftertaste,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictResponse(rating=rating)
```

---

## Step 6 — Create `Dockerfile.serving`

Create this file at the project root (same level as `Dockerfile.training`):

```dockerfile
# Coffee Text Analytics — FastAPI serving container
FROM python:3.9-slim

LABEL maintainer="Marcelo Seijas <marcelo.seijas@erasmusuniversity.nl>"
LABEL description="Coffee Analytics: FastAPI model serving"
LABEL version="1.0"

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy source and trained model artifacts
COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src

# BERT takes ~1-2 min to load — start-period must be long enough
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 7 — Update the `serve` target in `Makefile`

Find this block in `Makefile` (around line 158):
```makefile
serve: ## Start FastAPI serving layer (placeholder for future)
	@echo "$(BOLD)🚀 Starting API server...$(NC)"
	@echo "$(YELLOW)Note: FastAPI serving layer not yet implemented$(NC)"
	@echo "Coming in Task 4: Add FastAPI serving endpoint"
```

Replace with:
```makefile
serve: ## Start FastAPI serving layer on port 8000
	@echo "$(BOLD)Starting Coffee Analytics API on http://localhost:8000$(NC)"
	@echo "$(YELLOW)Requires: run 'make train' first to generate model artifacts$(NC)"
	cd src && uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

serve-docker: ## Build and run serving container
	docker build -f Dockerfile.serving -t coffee-serving:latest .
	docker run -p 8000:8000 coffee-serving:latest
```

---

## Step 8 — Create `tests/test_serving.py`

```python
"""
Tests for the FastAPI serving layer.

Unit tests mock the predictor to avoid loading BERT at test time.
Integration tests require real model artifacts (skipped in CI unless models/ is populated).
"""
import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODELS_DIR = Path(__file__).parent.parent / "models"
SERVING_READY = (MODELS_DIR / "xgboost_model.pkl").exists()


class TestServingEndpointsUnit(unittest.TestCase):
    """Unit tests — predictor is mocked, no BERT loading."""

    def setUp(self):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded = True
        mock_predictor.predict.return_value = 91.5

        # Patch at the module level before importing app
        patcher = patch("serving.predictor.CoffeePredictor", return_value=mock_predictor)
        patcher.start()
        self.addCleanup(patcher.stop)

        from fastapi.testclient import TestClient
        from serving.app import app
        self.client = TestClient(app)

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_model_info_returns_xgboost(self):
        resp = self.client.get("/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_name"], "xgboost")
        self.assertAlmostEqual(data["r2_score"], 0.9453, places=2)
        self.assertEqual(data["n_features_selected"], 279)

    def test_predict_missing_desc_1_returns_422(self):
        resp = self.client.post("/predict", json={})
        self.assertEqual(resp.status_code, 422)

    def test_predict_valid_request_returns_rating(self):
        resp = self.client.post("/predict", json={
            "desc_1": "bright citrus, clean finish",
            "desc_2": "lemon zest and floral notes",
            "desc_3": "exceptional clarity",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("rating", data)
        self.assertIsInstance(data["rating"], float)

    def test_predict_with_optional_sensory(self):
        resp = self.client.post("/predict", json={
            "desc_1": "bold and chocolatey",
            "aroma": 9.0,
            "acid": 7.5,
        })
        self.assertEqual(resp.status_code, 200)


@pytest.mark.skipif(not SERVING_READY, reason="model artifacts not trained yet")
@pytest.mark.integration
class TestServingIntegration(unittest.TestCase):
    """Integration tests — loads real model artifacts (slow, ~2 min startup)."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from serving.app import app
        cls.client = TestClient(app)

    def test_real_predict_in_range(self):
        resp = self.client.post("/predict", json={
            "desc_1": "jasmine tea, tangerine, brown sugar syrup",
            "desc_2": "Ethiopian natural process, light roast",
            "desc_3": "delicate and complex, exceptional value",
        })
        self.assertEqual(resp.status_code, 200)
        rating = resp.json()["rating"]
        self.assertGreaterEqual(rating, 80)
        self.assertLessEqual(rating, 100)


if __name__ == "__main__":
    unittest.main()
```

---

## Step 9 — Run formatting and lint

```bash
source ~/.virtualenvs/coffee-analytics/bin/activate
make format
make lint
```

Fix any issues before committing.

---

## Step 10 — Verify end-to-end

```bash
# 1. Start the server
source ~/.virtualenvs/coffee-analytics/bin/activate
make serve
# Wait ~2 min for BERT to load, watch for "Startup complete. API ready." in logs

# 2. In a second terminal — smoke test
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}

curl http://localhost:8000/model-info
# Expected: {"model_name":"xgboost","r2_score":0.9453,"n_features_selected":279,...}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"desc_1": "jasmine, tangerine, light roast", "desc_2": "Ethiopian natural", "desc_3": "exceptional clarity"}'
# Expected: {"rating": <float 80-100>, "model_name": "xgboost", "r2_score": 0.9453}

# 3. Run unit tests (fast, no BERT loading)
make test-one FILE=tests/test_serving.py
```

---

## Common issues and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'features'` | sys.path not set | Ensure `_SRC = Path(__file__).parent.parent` resolves to `src/` and is in sys.path |
| `FileNotFoundError: models/xgboost_model.pkl` | Pipeline not run | Run `make train` |
| `FileNotFoundError: models/tfidf_vectorizer.pkl` | Features step not run | Run `python main.py --steps features select train` |
| `RuntimeError: CoffeePredictor.load() must be called` | Lifespan not triggered | Use `TestClient(app)` as context manager or call `_predictor.load()` manually in tests |
| Port 8000 already in use | Previous process running | `lsof -ti:8000 | xargs kill` |
| BERT loading fails in Docker | Memory limit | Ensure container has ≥4GB RAM |
