from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

def load_model() -> Any:
    """
    Load the trained model artifact from disk.

    Returns:
        Trained scikit-learn pipeline.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python training/train.py` first."
        )

    return joblib.load(MODEL_PATH)