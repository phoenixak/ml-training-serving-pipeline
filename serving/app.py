"""FastAPI serving application for DLRM model predictions."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from serving.model_loader import DLRMModelLoader
from shared.config.serving_config import ServingConfig

logger = logging.getLogger(__name__)


# ---- Request/Response schemas ----


class PredictRequest(BaseModel):
    user_id: str
    item_id: str
    context: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    score: float
    cold_start: bool


class BatchPredictRequest(BaseModel):
    user_ids: List[str]
    item_ids: List[str]
    contexts: Optional[List[Dict[str, Any]]] = None


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class RecommendRequest(BaseModel):
    user_id: str
    num_recommendations: int = Field(default=10, ge=1, le=100)
    exclude_items: Optional[List[str]] = None


class RecommendationItem(BaseModel):
    item_id: str
    score: float
    rank: int


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]


# ---- Application setup ----

_model_loader: Optional[DLRMModelLoader] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model_loader
    config = ServingConfig()
    model_dir = os.environ.get("MODEL_DIR", config.model_dir)
    _model_loader = DLRMModelLoader(config)
    loaded = _model_loader.load_from_training_output(model_dir)
    if loaded:
        logger.info("Model loaded successfully on startup")
    else:
        logger.warning(
            "Model not available at startup; serving will return errors until a model is loaded"
        )
    yield
    _model_loader = None


app = FastAPI(
    title="DLRM Model Serving API",
    description="FastAPI application for DLRM model predictions and recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_loader() -> DLRMModelLoader:
    """Return the model loader, raising 404 if not loaded."""
    if _model_loader is None or not _model_loader.is_loaded():
        raise HTTPException(status_code=404, detail="Model not loaded")
    return _model_loader


# ---- Endpoints ----


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return model loaded status and model info."""
    if _model_loader is None:
        return HealthResponse(
            status="unavailable",
            model_loaded=False,
            model_info={},
        )
    return HealthResponse(
        status="ok" if _model_loader.is_loaded() else "no_model",
        model_loaded=_model_loader.is_loaded(),
        model_info=_model_loader.get_model_info(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate a single prediction for a user-item pair."""
    loader = _get_loader()
    user_idx = loader.user_mapping.get(request.user_id)
    item_idx = loader.item_mapping.get(request.item_id)
    cold_start = user_idx is None or item_idx is None
    score = loader.predict_single(request.user_id, request.item_id, request.context)
    return PredictResponse(score=score, cold_start=cold_start)


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Generate predictions for multiple user-item pairs."""
    loader = _get_loader()
    if len(request.user_ids) != len(request.item_ids):
        raise HTTPException(
            status_code=422,
            detail="user_ids and item_ids must have the same length",
        )
    scores = loader.predict_batch(request.user_ids, request.item_ids, request.contexts)
    predictions = []
    for i, score in enumerate(scores):
        user_idx = loader.user_mapping.get(request.user_ids[i])
        item_idx = loader.item_mapping.get(request.item_ids[i])
        cold_start = user_idx is None or item_idx is None
        predictions.append(PredictResponse(score=score, cold_start=cold_start))
    return BatchPredictResponse(predictions=predictions)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Generate top-k item recommendations for a user."""
    loader = _get_loader()
    results = loader.recommend_items(
        user_id=request.user_id,
        num_recommendations=request.num_recommendations,
        exclude_items=request.exclude_items,
    )
    recommendations = [
        RecommendationItem(
            item_id=r["item_id"],
            score=r["score"],
            rank=r["rank"],
        )
        for r in results
    ]
    return RecommendResponse(recommendations=recommendations)


if __name__ == "__main__":
    uvicorn.run(
        "serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
