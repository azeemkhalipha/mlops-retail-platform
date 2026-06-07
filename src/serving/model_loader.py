import os
import pickle
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT  = os.getenv("PROJECT_ROOT", "/app")
MODEL_PATH    = f"{PROJECT_ROOT}/models/demand_forecast_v1.pkl"
MODEL_VERSION = "1"


class ModelLoader:
    """
    Handles loading the ML model from disk and making predictions.
    Kept as a class so the model is loaded once at startup
    and reused for every request — not reloaded on every call.
    """

    def __init__(self):
        self.model         = None
        self.model_version = None
        self.is_loaded     = False

    def load(self):
        """Load model from pickle file into memory."""
        try:
            print(f"Loading model from: {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            self.model_version = MODEL_VERSION
            self.is_loaded     = True
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    def predict(self, features: dict) -> float:
        """Make a single prediction from a dict of feature values."""
        import pandas as pd
        df  = pd.DataFrame([features])
        out = self.model.predict(df)
        return round(float(out[0]), 2)

    def batch_predict(self, features_list: list) -> list:
        """Make predictions for a list of feature dicts."""
        import pandas as pd
        df   = pd.DataFrame(features_list)
        outs = self.model.predict(df)
        return [round(float(v), 2) for v in outs]


# Singleton — one instance shared across all API requests
model_loader = ModelLoader()
