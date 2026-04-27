from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum
import numpy as np
import pandas as pd

# ==========================================
# 1. INITIALIZE API & LOAD AI INTO RAM (Cold Start)
# ==========================================
app = FastAPI(title="PJMW Grid Load Forecaster", version="2.0")

print("Loading optimized RBF model and parameters into AWS memory...")
try:
    # Load scaling parameters
    params = np.load("processed_data/scaling_params.npz")
    MIN_MW = params['min_value']
    MAX_MW = params['max_value']

    # Load the 29-feature champion model
    model = np.load("processed_data/rbf_model_opt.npz")
    CENTERS = model['centers']
    SIGMAS = model['sigma']
    WEIGHTS = model['weights']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR loading model files: {e}")

# ==========================================
# 2. DEFINE THE API JSON INPUT SCHEMA
# ==========================================
class PredictionRequest(BaseModel):
    past_24_hours: list[float]  # A list of exactly 24 MW values
    lag_168_mw: float           # The MW value from exactly one week ago
    target_datetime: str        # The exact hour you want to predict (e.g., "2026-03-06 14:00:00")

# ==========================================
# 3. THE PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
async def predict_megawatts(request: PredictionRequest):
    # --- SAFETY CHECKS ---
    if len(request.past_24_hours) != 24:
        raise HTTPException(status_code=400, detail="Must provide exactly 24 hours of historical data.")
    
    try:
        # --- STEP 1: PARSE DATETIME & CALCULATE CYCLICAL TIME ---
        dt = pd.to_datetime(request.target_datetime)
        hour = dt.hour
        day = dt.dayofweek

        # The Trigonometry features (Features 26, 27, 28, 29)
        hour_sin = np.sin(hour * (2. * np.pi / 24))
        hour_cos = np.cos(hour * (2. * np.pi / 24))
        day_sin = np.sin(day * (2. * np.pi / 7))
        day_cos = np.cos(day * (2. * np.pi / 7))

        # --- STEP 2: MIN-MAX SCALING ---
        def scale_value(val):
            return (val - MIN_MW) / (MAX_MW - MIN_MW)

        # Scale the 24 hours and the 1-week lag (Features 1-25)
        scaled_24 = [scale_value(mw) for mw in request.past_24_hours]
        scaled_lag = scale_value(request.lag_168_mw)

        # --- STEP 3: ASSEMBLE THE 29-DIMENSIONAL VECTOR ---
        # We glue them all together exactly as we did in the training script
        X_input = np.array([
            *scaled_24, 
            scaled_lag, 
            hour_sin, hour_cos, 
            day_sin, day_cos
        ])

        # --- STEP 4: THE RBF INFERENCE MATH ---
        # 1. Calculate distance from this 1 input to all K centers
        distances = np.linalg.norm(X_input - CENTERS, axis=1)
        
        # 2. Pass through the Gaussian Hidden Layer (using the unique sigmas)
        G = np.exp(-(distances ** 2) / (2 * SIGMAS ** 2))
        
        # 3. Multiply by weights to get the final scaled prediction
        y_pred_scaled = np.dot(G, WEIGHTS)

        # --- STEP 5: REVERSE SCALING ---
        y_pred_real = (y_pred_scaled * (MAX_MW - MIN_MW)) + MIN_MW

        # Return the final business-ready response
        return {
            "target_datetime": request.target_datetime,
            "predicted_megawatts": round(float(y_pred_real), 2),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==========================================
# 4. AWS LAMBDA WRAPPER
# ==========================================
# Mangum packages the FastAPI app so AWS Lambda can trigger it
handler = Mangum(app)