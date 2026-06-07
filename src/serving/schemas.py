from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """
    Defines what a single prediction request must contain.
    Every field is required and must be a float.
    Pydantic automatically validates this — wrong types get rejected.
    """
    qty_lag_1:           float = Field(..., description="Quantity sold 1 day ago")
    qty_lag_7:           float = Field(..., description="Quantity sold 7 days ago")
    qty_lag_30:          float = Field(..., description="Quantity sold 30 days ago")
    qty_rolling_avg_7:   float = Field(..., description="7-day rolling average")
    qty_rolling_avg_30:  float = Field(..., description="30-day rolling average")
    qty_rolling_std_7:   float = Field(..., description="7-day rolling std deviation")
    daily_revenue:       float = Field(..., description="Revenue on that day")

    class Config:
        json_schema_extra = {
            "example": {
                "qty_lag_1":          10.0,
                "qty_lag_7":          8.0,
                "qty_lag_30":         9.0,
                "qty_rolling_avg_7":  9.5,
                "qty_rolling_avg_30": 9.0,
                "qty_rolling_std_7":  1.2,
                "daily_revenue":      45.0
            }
        }


class PredictionResponse(BaseModel):
    """What the API returns for a single prediction."""
    predicted_quantity: float
    model_version:      str
    status:             str


class BatchPredictionRequest(BaseModel):
    """A list of prediction requests sent together."""
    inputs: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """What the API returns for a batch prediction."""
    predictions: List[float]
    count:       int
    status:      str


class HealthResponse(BaseModel):
    """API health check response."""
    status:        str
    model_loaded:  bool
    model_version: str
