from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model_loader import load_model


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 numeric features for one sample.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model once when the API starts.
    """
    app.state.model = load_model()
    yield


app = FastAPI(
    title="MLOps Model Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = app.state.model

    try:
        input_array = np.array(request.features).reshape(1, -1)

        prediction = int(model.predict(input_array)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_array)[0].max())
        else:
            probability = 1.0

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
        )

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error