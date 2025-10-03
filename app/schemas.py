from typing import Dict, List
from pydantic import BaseModel, validator

from .constants import FEATURE_COLUMNS

class PredictionRequest(BaseModel):
    data: List[List[float]]

    @validator("data")
    def validate_numeric(cls, v):
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Data must be a list of lists")
        for i, row in enumerate(v):
            if len(row) != len(FEATURE_COLUMNS):
                raise ValueError(f"Row {i} must have {len(FEATURE_COLUMNS)} features")
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Value at row {i}, col {j} is not numeric")
        return v

class PredictionResponse(BaseModel):
    models: Dict[str, dict]

class ModelInfoResponse(BaseModel):
    model: str
    features_count: int
    model_path: str
    timestamp: str
    file_size_bytes: int
    model_exists: bool

class ModelHealth(BaseModel):
    status: str
    file_size_bytes: int

class HealthResponse(BaseModel):
    status: str
    models: Dict[str, ModelHealth]