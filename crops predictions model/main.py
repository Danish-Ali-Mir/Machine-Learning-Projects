from fastapi.responses import FileResponse
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "crop_model.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = FastAPI(
    title="Crop Recommendation API",
    description="Recommends the best crop to grow based on soil and weather conditions.",
    version="1.0.0",
)

# Load the trained model once, when the server starts
if not MODEL_PATH.exists():
    raise RuntimeError("crop_model.joblib not found. Run `python train_model.py` first.")
model = joblib.load(MODEL_PATH)


class CropInput(BaseModel):
    N: float = Field(..., ge=0, description="Nitrogen content in soil", examples=[90])
    P: float = Field(..., ge=0, description="Phosphorus content in soil", examples=[42])
    K: float = Field(..., ge=0, description="Potassium content in soil", examples=[43])
    temperature: float = Field(..., description="Temperature in °C", examples=[20.9])
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in %", examples=[82.0])
    ph: float = Field(..., ge=0, le=14, description="Soil pH value", examples=[6.5])
    rainfall: float = Field(..., ge=0, description="Rainfall in mm", examples=[202.9])


class CropOutput(BaseModel):
    recommended_crop: str
    confidence: float
    top_3: list[dict]


@app.get("/")
def home():
    return FileResponse("template/index.html")


@app.post("/predict", response_model=CropOutput)
def predict(payload: CropInput):
    try:
        # DataFrame keeps column names consistent with training
        row = pd.DataFrame([[getattr(payload, f) for f in FEATURES]], columns=FEATURES)
        probs = model.predict_proba(row)[0]
        top_idx = probs.argsort()[::-1][:3]

        top_3 = [
            {"crop": str(model.classes_[i]), "probability": round(float(probs[i]), 4)}
            for i in top_idx
        ]
        return CropOutput(
            recommended_crop=top_3[0]["crop"],
            confidence=top_3[0]["probability"],
            top_3=top_3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
