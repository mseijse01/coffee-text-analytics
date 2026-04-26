"""Pydantic models for FastAPI serving layer."""

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for /predict endpoint."""

    desc_1: str = Field(..., description="Primary tasting notes")
    desc_2: str = Field("", description="Secondary review notes")
    desc_3: str = Field("", description="Bottom-line conclusion")
    roast: str = Field("", description="Roast level (e.g. 'Light', 'Medium', 'Dark')")
    country_of_origin: str = Field(
        "", description="Country of origin (e.g. 'Ethiopia')"
    )
    roaster: str = Field("", description="Roaster name")
    aroma: Optional[float] = Field(None, ge=0, le=10)
    acid: Optional[float] = Field(None, ge=0, le=10)
    body: Optional[float] = Field(None, ge=0, le=10)
    flavor: Optional[float] = Field(None, ge=0, le=10)
    aftertaste: Optional[float] = Field(None, ge=0, le=10)


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""

    rating: float = Field(..., description="Predicted coffee rating (80–100 scale)")
    model_name: str = "xgboost"
    r2_score: float = 0.9453


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response schema for /model-info endpoint."""

    model_name: str
    r2_score: float
    n_features_selected: int
    feature_extractors: list
