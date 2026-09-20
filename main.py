"""
FastAPI serving layer for the AI4I Predictive Maintenance model.

Loads the saved XGBoost model and decision threshold, then exposes
endpoints for machine failure prediction.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Load saved artifacts
# ============================================================

MODEL_PATH = "xgb_final_model.pkl"
THRESHOLD_PATH = "xgb_final_threshold.pkl"
FEATURES_PATH = "feature_names.pkl"

try:
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    feature_names = joblib.load(FEATURES_PATH)

except FileNotFoundError as e:
    raise RuntimeError(
        f"Could not load model artifacts: {e}. "
        "Make sure xgb_final_model.pkl, "
        "xgb_final_threshold.pkl and feature_names.pkl "
        "are available."
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="AI4I 2020 machine failure prediction API using XGBoost",
    version="1.0.0",
)


# ============================================================
# Request schema
# ============================================================

class SensorReading(BaseModel):

    type: str = Field(
        ...,
        description="Machine quality variant: L, M, or H"
    )

    air_temperature_k: float = Field(
        ...,
        description="Air temperature [K]"
    )

    process_temperature_k: float = Field(
        ...,
        description="Process temperature [K]"
    )

    rotational_speed_rpm: float = Field(
        ...,
        description="Rotational speed [rpm]"
    )

    torque_nm: float = Field(
        ...,
        description="Torque [Nm]"
    )

    tool_wear_min: float = Field(
        ...,
        description="Tool wear [min]"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "type": "M",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0
            }
        }


# ============================================================
# Response schema
# ============================================================

class PredictionResponse(BaseModel):

    failure_probability: float
    prediction: int
    threshold_used: float


# ============================================================
# Preprocessing
# ============================================================

def preprocess(reading: SensorReading) -> pd.DataFrame:

    machine_type = reading.type.upper()

    # --------------------------------------------------------
    # Validate machine type
    # --------------------------------------------------------

    if machine_type not in {"L", "M", "H"}:
        raise HTTPException(
            status_code=400,
            detail="type must be one of L, M, H"
        )

    # --------------------------------------------------------
    # Feature engineering
    # Same calculations used during model training
    # --------------------------------------------------------

    temperature_difference = (
        reading.process_temperature_k
        - reading.air_temperature_k
    )

    mech_power = (
        reading.torque_nm
        * reading.rotational_speed_rpm
        / 9.549
    )

    # --------------------------------------------------------
    # One-hot encoding for Type
    # Same representation used during training
    # --------------------------------------------------------

    row = {

        "Air temperature [K]":
            reading.air_temperature_k,

        "Process temperature [K]":
            reading.process_temperature_k,

        "Rotational speed [rpm]":
            reading.rotational_speed_rpm,

        "Torque [Nm]":
            reading.torque_nm,

        "Tool wear [min]":
            reading.tool_wear_min,

        "Temperature Difference":
            temperature_difference,

        "MechPower":
            mech_power,

        "Type_H":
            1 if machine_type == "H" else 0,

        "Type_L":
            1 if machine_type == "L" else 0,

        "Type_M":
            1 if machine_type == "M" else 0,
    }

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([row])

    # --------------------------------------------------------
    # Verify that all expected model features exist
    # --------------------------------------------------------

    missing_features = set(feature_names) - set(df.columns)

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=(
                "Preprocessing is missing expected features: "
                f"{missing_features}"
            )
        )

    # --------------------------------------------------------
    # Enforce EXACT training feature order
    # --------------------------------------------------------

    df = df[feature_names]

    return df


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "threshold": threshold
    }


# ============================================================
# Single prediction
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(reading: SensorReading):

    X = preprocess(reading)

    # Get probability of machine failure
    probability = float(
        model.predict_proba(X)[0][1]
    )

    # Apply validated threshold
    prediction = int(
        probability >= threshold
    )

    return PredictionResponse(
        failure_probability=probability,
        prediction=prediction,
        threshold_used=threshold
    )


# ============================================================
# Batch prediction
# ============================================================

@app.post("/predict/batch")
def predict_batch(
    readings: list[SensorReading]
):

    results = []

    for reading in readings:

        X = preprocess(reading)

        probability = float(
            model.predict_proba(X)[0][1]
        )

        prediction = int(
            probability >= threshold
        )

        results.append({
            "failure_probability": probability,
            "prediction": prediction,
            "threshold_used": threshold
        })

    return results