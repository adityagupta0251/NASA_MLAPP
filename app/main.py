import os
import numpy as np
from fastapi import FastAPI, HTTPException, Query

from .constants import FEATURE_COLUMNS, MODEL_PATHS
from .utils import predict, get_model_info
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    ModelHealth
)

app = FastAPI(title="AstroCluster Exoplanet ML API", version="1.1")

@app.get("/", summary="API Root")
def root():
    return {
        "message": "AstroCluster Exoplanet ML API",
        "version": app.version,
        "endpoints": ["/health","/features","/model_info","/model/{model_type}/info","/predict"]
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    models_status = {
        m: ModelHealth(
            status="healthy" if os.path.exists(path) else "missing",
            file_size_bytes=os.path.getsize(path) if os.path.exists(path) else 0
        )
        for m, path in MODEL_PATHS.items()
    }
    return HealthResponse(status="healthy", models=models_status)

@app.get("/features", summary="List Features")
def list_features():
    return {"feature_names": FEATURE_COLUMNS, "feature_count": len(FEATURE_COLUMNS)}

@app.get("/model_info", response_model=list[ModelInfoResponse], summary="All Models Info")
def list_models_info():
    return [get_model_info(m) for m in MODEL_PATHS]

@app.get("/model/{model_type}/info", response_model=ModelInfoResponse, summary="Single Model Info")
def single_model_info(model_type: str):
    mt = model_type.lower()
    if mt not in MODEL_PATHS:
        raise HTTPException(404, f"Model '{model_type}' not found")
    return get_model_info(mt)

@app.post("/predict", response_model=PredictionResponse, summary="Predict Exoplanet Candidate")
def predict_multiple_models(
    req: PredictionRequest,
    model_types: list[str] = Query(default=["rf","xgb","lr"])
):
    mts = [m.lower() for m in model_types]
    invalid = [m for m in mts if m not in MODEL_PATHS]
    if invalid:
        raise HTTPException(400, f"Invalid models: {invalid}")

    X = np.array(req.data)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
        raise HTTPException(400, f"Each sample must have {len(FEATURE_COLUMNS)} features")

    results = {m: dict(zip(["predictions","probabilities"], predict(m, X)))
               for m in mts}
    return PredictionResponse(models=results)
