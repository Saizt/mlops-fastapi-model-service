import os
from pathlib import Path
from typing import Any

import joblib
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def load_model() -> Any:
    """
    Load a model either from MLflow or from the local model artifact.

    Default behavior:
        Load models/model.joblib.

    MLflow behavior:
        If MLFLOW_MODEL_URI is set, load the model from MLflow.
    """
    mlflow_model_uri = os.getenv("MLFLOW_MODEL_URI")

    if mlflow_model_uri:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        return mlflow.pyfunc.load_model(mlflow_model_uri)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python training/train.py` first."
        )

    return joblib.load(MODEL_PATH)