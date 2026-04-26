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
    """Load ML artifacts on startup.

    BERT takes ~1-2 min — start-period in Docker is set accordingly.
    """
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
    """Health check endpoint."""
    return HealthResponse(status="ok", model_loaded=_predictor.is_loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return model metadata."""
    return ModelInfoResponse(
        model_name="xgboost",
        r2_score=0.9453,
        n_features_selected=279,
        feature_extractors=["tfidf", "bert", "topics", "sentiment"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict coffee rating from review text."""
    if not _predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    try:
        rating = _predictor.predict(
            desc_1=request.desc_1,
            desc_2=request.desc_2,
            desc_3=request.desc_3,
            roast=request.roast,
            country_of_origin=request.country_of_origin,
            roaster=request.roaster,
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
