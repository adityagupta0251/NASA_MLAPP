from typing import Dict, List
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    data: List[List[float]]

    @validator("data")
    def validate_numeric(cls, v):
        # Ensure data is a list of lists
        if not isinstance(v, list) or not all(isinstance(row, list) for row in v):
            raise ValueError("Data must be a list of lists")
        # Ensure all values are numeric
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Value at row {i}, column {j} is not numeric: {val}")
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
