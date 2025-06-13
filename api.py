"""
api.py

🚖 NYC Taxi Demand Forecasting API
----------------------------------
This FastAPI module handles model training requests for forecasting taxi demand
based on weekday and time intervals. It processes dataset selection, preprocessing,
LSTM model training, and saves the trained components.

Endpoints:
- GET "/"               : Health check endpoint.
- POST "/train_model/" : Trains and saves model for a given weekday and time interval.

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from data import preprocess_data, create_sequences
from model import build_and_train_model, save_model_components
from config import WEEKDAYS, DATASET_FOLDER
import secrets

app = FastAPI()

# --- API Key Setup ---
# 1. Try to read API key from API_KEY.txt (for easy management, not for production sharing!)
# 2. If not found, try environment variable TAXI_API_KEY (recommended for production)
# 3. If neither, generate a secure random key and print it (for local/dev only)

API_KEY = None
api_key_path = os.path.join(os.path.dirname(__file__), "API_KEY.txt")
print(f"[DEBUG] Looking for API_KEY.txt at: {api_key_path}")
if os.path.exists(api_key_path):
    with open(api_key_path, "r") as f:
        file_content = f.read()
        print(f"[DEBUG] Raw content from API_KEY.txt: '{file_content}'")
        API_KEY = file_content.strip()
        if API_KEY:
            print("[INFO] API key loaded from API_KEY.txt")
        else:
            print("[WARN] API_KEY.txt is empty after stripping whitespace!")
if not API_KEY:
    API_KEY = os.environ.get("TAXI_API_KEY")
    if API_KEY:
        print("[INFO] API key loaded from TAXI_API_KEY environment variable")
if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    print(f"[INFO] No API_KEY.txt or TAXI_API_KEY env var set. Using generated API key: {API_KEY}")

# --- API Key verification ---
def verify_api_key():
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not properly configured")

class TrainRequest(BaseModel):
    weekday_num: int
    start_hour: int
    end_hour: int
    am_pm_start: str
    am_pm_end: str

@app.get("/")
def read_root():
    return {"message": "🚀 FastAPI is working!"}

@app.post("/train_model/")
async def train_model(request: TrainRequest):
    verify_api_key()

    # Validate weekday input
    try:
        weekday_name = WEEKDAYS[request.weekday_num]
    except IndexError:
        return {"error": "❌ Invalid weekday_num. Must be between 0 (Monday) and 6 (Sunday)."}
    # Convert to 24-hour format
    start_hour_24 = request.start_hour
    end_hour_24 = request.end_hour
    if request.am_pm_start.upper() == "PM" and start_hour_24 < 12:
        start_hour_24 += 12
    if request.am_pm_end.upper() == "PM" and end_hour_24 < 12:
        end_hour_24 += 12
    # Build dataset path
    file_name = f"{weekday_name}_{request.start_hour}_{request.am_pm_start}_to_{request.end_hour}_{request.am_pm_end}_data.csv"
    file_path = os.path.join(DATASET_FOLDER, file_name)
    if not os.path.exists(file_path):
        return {"error": f"❌ Dataset file not found: {file_name}"}
    try:
        agg_df, le = preprocess_data(file_path, start_hour_24, end_hour_24, request.weekday_num)
        X, y, hex_ids, _ = create_sequences(agg_df)
        model, scaler = build_and_train_model(X, y, hex_ids, agg_df)
        file_prefix = f"{weekday_name}_{request.start_hour}_{request.am_pm_start}_to_{request.end_hour}_{request.am_pm_end}"
        save_model_components(model, scaler, le, file_prefix)
        return {
            "message": "✅ Model trained and saved!",
            "files_saved": [
                f"{file_prefix}_model.keras",
                f"{file_prefix}_scaler.pkl",
                f"{file_prefix}_labelencoder.pkl"
            ],
            "model_prefix": file_prefix
        }
    except Exception as e:
        return {"error": f"❌ Model training failed: {str(e)}"}
