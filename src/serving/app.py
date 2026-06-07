import sys
import os

# Add serving folder to Python path so local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from schemas import (  # noqa: E402
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from model_loader import model_loader  # noqa: E402

FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_30",
    "qty_rolling_avg_7", "qty_rolling_avg_30",
    "qty_rolling_std_7", "daily_revenue"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Model is loaded here — once at startup, not on every request.
    """
    print("API starting up — loading model...")
    model_loader.load()
    yield  # API runs here
    print("API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Retail Demand Forecast API",
    description="MLOps platform for retail demand forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Allow requests from any origin (needed for browser-based clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "Retail Demand Forecast API is running",
        "docs":    "/docs",
        "health":  "/health"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Used by Docker, CI, and monitoring tools to verify the API is alive.
    Returns whether the model is loaded and ready to serve predictions.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded,
        model_version=model_loader.model_version or "none"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Single prediction endpoint.
    Accepts one set of features, returns one predicted quantity.
    """
    try:
        # Extract feature values from request in the correct order
        features   = {col: getattr(request, col) for col in FEATURE_COLS}
        prediction = model_loader.predict(features)

        return PredictionResponse(
            predicted_quantity=prediction,
            model_version=model_loader.model_version,
            status="success"
        )
    except Exception as e:
        # Return 500 with error detail if prediction fails
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.
    Accepts multiple sets of features, returns multiple predictions.
    More efficient than calling /predict multiple times.
    """
    try:
        features_list = [
            {col: getattr(item, col) for col in FEATURE_COLS}
            for item in request.inputs
        ]
        predictions = model_loader.batch_predict(features_list)

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
