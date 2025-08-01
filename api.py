from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Literal
import torch
import torch.nn.functional as F
import logging
import os
import pandas as pd
import numpy as np
import uvicorn
from cmodel_1 import FloodNet, ord_enc, text_enc, onehot_enc, scaler, valid_categories  # Import encoders and categories

# Initialize app
app = FastAPI(title="Flood Risk Prediction API")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = os.getenv("MODEL_PATH", "best_floodnet_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found")

model = FloodNet().to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load dataset for preprocessing parameters
csv_path = os.getenv("DATA_PATH", "flood_risk_dataset_final.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset file {csv_path} not found")
df = pd.read_csv(csv_path)

# Feature definitions
text_feature = "ProximityToWaterBody"
numerical_features = ["Elevation", "Rainfall"]
ordinal_features = [
    "Vegetation", "Urbanization", "Drainage", "Slope", "StormFrequency",
    "Deforestation", "Infrastructure", "Encroachment", "Season"
]
onehot_features = ["Soil", "Wetlands"]
class_labels = ["None", "Low", "Medium", "High"]  # Adjust based on your LabelEncoder

# Input schema
class FloodFeatures(BaseModel):
    Vegetation: Literal["Missing", "Sparse", "Moderate", "Dense"]
    Urbanization: Literal["Low", "Medium", "High"]
    Drainage: Literal["Poor", "Moderate", "Good"]
    Slope: Literal["Flat", "Moderate", "Steep"]
    StormFrequency: Literal["Rare", "Occasional", "Frequent"]
    Deforestation: Literal["Missing", "Moderate", "Severe"]
    Infrastructure: Literal["Weak", "Moderate", "Strong"]
    Encroachment: Literal["Missing", "Moderate", "Severe"]
    Season: Literal["Dry", "Rainy", "Transition"]
    Soil: Literal["Clay", "Sandy", "Loamy", "Rocky"]
    Wetlands: Literal["Present", "Absent"]
    ProximityToWaterBody: Literal[
        "Close to Dam", "Close to Lagoon", "Close to Lake", "Close to River", "Close to Stream",
        "Far from Dam", "Far from Lagoon", "Far from Lake", "Far from River", "Far from Stream",
        "Moderately Close to Dam", "Moderately Close to Lagoon", "Moderately Close to Lake",
        "Moderately Close to River", "Moderately Close to Stream",
        "Very Close to Dam", "Very Close to Lagoon", "Very Close to Lake", "Very Close to River", "Very Close to Stream",
        "Very Far from Dam", "Very Far from Lagoon", "Very Far from Lake", "Very Far from River", "Very Far from Stream"
    ]
    Rainfall: float
    Elevation: float

    @validator("Rainfall", "Elevation")
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Must be non-negative")
        return v

# Output schema
class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict

# Preprocessing function
def preprocess_input(features: FloodFeatures):
    try:
        # Create DataFrame from input
        user_df = pd.DataFrame([features.dict()])
        
        # Handle missing values
        for col in ordinal_features + onehot_features + [text_feature]:
            user_df[col] = user_df[col].fillna("Missing")
        
        # Encode features
        text_tensor = torch.tensor(
            text_enc.transform(user_df[[text_feature]]).astype(np.int64).squeeze(1),
            dtype=torch.long
        ).to(device)
        
        ordinal_tensor = torch.tensor(
            ord_enc.transform(user_df[ordinal_features]).astype(np.int64),
            dtype=torch.long
        ).to(device)
        
        onehot_tensor = torch.tensor(
            onehot_enc.transform(user_df[onehot_features]).astype(np.float32),
            dtype=torch.float
        ).to(device)
        
        # Process numerical features
        numerical = user_df[numerical_features].copy()
        for col in numerical_features:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            numerical[col] = numerical[col].clip(lower=max(q1 - 1.5 * iqr, 0), upper=q3 + 1.5 * iqr)
            mn = numerical[col].min()
            numerical[col] = np.log1p(numerical[col] + abs(mn) + 1e-6)
        
        num_tensor = torch.tensor(
            scaler.transform(numerical).astype(np.float32),
            dtype=torch.float
        ).to(device)
        
        return text_tensor, ordinal_tensor, onehot_tensor, num_tensor
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: FloodFeatures):
    logger.info(f"Received input: {features.dict()}")

    try:
        # Preprocess input
        text_tensor, ordinal_tensor, onehot_tensor, num_tensor = preprocess_input(features)
        
        # Predict
        with torch.no_grad():
            output = model(text_tensor, ordinal_tensor, onehot_tensor, num_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            probs_dict = {
                label:f"{round(float(probs[0][i])*100,4)}%" for i,label in enumerate(class_labels)
            }
        
        return PredictionResponse(
            predicted_class=class_labels[pred_idx],
            probabilities=probs_dict
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/valid_options")
async def get_valid_options():
    return valid_categories

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=9000)


    




