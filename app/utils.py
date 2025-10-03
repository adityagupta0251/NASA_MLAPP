import os
import pickle
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import MODEL_PATHS, FEATURE_COLUMNS

# Cache for loaded models
_loaded_models: Dict[str, object] = {}

def load_model(model_type: str):
    """Load and cache a model from disk."""
    model_type = model_type.lower()
    if model_type in _loaded_models:
        return _loaded_models[model_type]

    path = MODEL_PATHS.get(model_type)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model '{model_type}' not found at {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)
    _loaded_models[model_type] = model
    return model

def predict(model_type: str, X: np.ndarray) -> Tuple[List[int], List[float]]:
    """Run predictions and return binary labels plus probabilities."""
    model_type = model_type.lower()
    model = load_model(model_type)

    df_X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    if model_type == "xgb":
        dmatrix = xgb.DMatrix(df_X)
        probs = model.predict(dmatrix).tolist()
    else:
        probs = model.predict_proba(df_X)[:, 1].tolist()

    preds = [1 if p > 0.5 else 0 for p in probs]
    return preds, probs

def get_model_info(model_type: str) -> dict:
    """Return metadata about a given model."""
    model_type = model_type.lower()
    path = MODEL_PATHS.get(model_type)
    if not path:
        raise ValueError(f"Unknown model type: {model_type}")

    exists = os.path.exists(path)
    return {
        "model": model_type,
        "features_count": len(FEATURE_COLUMNS),
        "model_path": path,
        "timestamp": datetime.now().isoformat(),
        "file_size_bytes": os.path.getsize(path) if exists else 0,
        "model_exists": exists
    }
