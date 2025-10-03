import os
import numpy as np
from fastapi import FastAPI, HTTPException, Query

from .constants import FEATURE_COLUMNS, MODEL_PATHS
from .utils import predict, get_model_info
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse, ModelHealth
)

app = FastAPI(title="AstroCluster Exoplanet ML API", version="1.1")

@app.get("/")
def root():
    return {
        "message": "AstroCluster Exoplanet ML API",
        "version": app.version,
        "endpoints": {
            "/features": "List required features",
            "/model_info": "Info on all models",
            "/model/{model_type}/info": "Info on a single model",
            "/predict": "Generate predictions",
            "/health": "Health check"
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Service and model health check")
def health_check():
    models_status = {}
    for m, path in MODEL_PATHS.items():
        exists = os.path.exists(path)
        models_status[m] = ModelHealth(
            status="healthy" if exists else "missing",
            file_size_bytes=os.path.getsize(path) if exists else 0
        )
    return HealthResponse(status="healthy", models=models_status)

@app.get("/features")
def list_features():
    return {
        "feature_names": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS)
    }

@app.get("/model_info", response_model=list[ModelInfoResponse])
def list_models_info():
    return [get_model_info(m) for m in MODEL_PATHS.keys()]

@app.get("/model/{model_type}/info", response_model=ModelInfoResponse)
def single_model_info(model_type: str):
    mt = model_type.lower()
    if mt not in MODEL_PATHS:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")
    return get_model_info(mt)

@app.post("/predict", response_model=PredictionResponse)
def predict_multiple_models(
    req: PredictionRequest,
    model_types: list[str] = Query(default=["rf", "xgb", "lr"])
):
    mts = [m.lower() for m in model_types]
    invalid = [m for m in mts if m not in MODEL_PATHS]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid model types: {invalid}")

    X = np.array(req.data)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
        raise HTTPException(
            status_code=400,
            detail=f"Each sample must have {len(FEATURE_COLUMNS)} features"
        )

    results = {}
    for m in mts:
        preds, probs = predict(m, X)
        results[m] = {"predictions": preds, "probabilities": probs}

    # ✅ Remove trailing comma to avoid returning a tuple
    return PredictionResponse(models=results)
