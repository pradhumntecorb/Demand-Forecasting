# Project_Readme

## NYC Taxi Demand Forecasting System — Project Overview

This project provides a modular, production-ready, and scalable pipeline for forecasting NYC taxi demand using LSTM-based time series models, with optional reinforcement learning for driver dispatch. It includes a REST API (FastAPI) and an interactive UI (Streamlit).

---

## Workflow Chart (with Security & API Key Management)

```mermaid
flowchart TD
    A[User/Streamlit UI/API] --> B[Data Preparation<br>(data.py)]
    B --> C[Model Training<br>(model.py via API/UI)]
    C --> D[Model Saving]
    D --> E[Demand Prediction<br>(predict.py via UI)]
    E --> F[Visualization<br>(Streamlit UI)]
    E --> G{Taxi Allocation?<br>(Optional)}
    G -- Yes --> H[Allocate Taxis]
    G -- No --> F
    H --> F
    %% Security additions
    subgraph Security
        S1[API Key Management<br>API_KEY.txt or TAXI_API_KEY env var] --> S2[.gitignore protects API_KEY.txt]
        S2 --> S3[API endpoints require x-api-key header]
    end
    A -.-> S1
    C -.-> S3
```

**Key Security & Best Practices Integrated:**
- API key is loaded from `API_KEY.txt` (local/dev) or the `TAXI_API_KEY` environment variable (production).
- `.gitignore` ensures API keys and sensitive files are never committed to git.
- All sensitive API endpoints (e.g., `/train_model/`) require the correct `x-api-key` header for access.
- Debugging output helps verify which key source is used and prevents accidental misconfiguration.

---

## Security & API Key Management

- **API Key Loading:**
  - Loads from `API_KEY.txt` (for local/dev; file is gitignored)
  - Or from the `TAXI_API_KEY` environment variable (for production)
  - If neither is set, generates a secure random key for local/dev sessions
- **.gitignore:**
  - Ensures secrets like `API_KEY.txt` are never committed to version control
- **API Endpoint Protection:**
  - All sensitive endpoints (e.g., `/train_model/`) require the correct API key in the `x-api-key` header
- **Debugging:**
  - Startup debug logs show where the API key was loaded from and warn if the file is empty or misconfigured

---

## Project Structure

project_root/
│
├── config.py               # ✅ Central config: paths, constants, global settings
├── requirements.txt        # ✅ Project dependencies (FastAPI, Streamlit, Keras, etc.)
├── Project_Readme.md       # ✅ Complete project documentation (detailed instructions)
├── .gitignore              # 🚫 Ignore secrets (API_KEY.txt) and local files
├── API_KEY.txt             # 🔑 (gitignored) Stores API key for local/dev use
│
├── datasets/               # 📂 Raw & preprocessed input data (per time interval)
│   └── *.csv               # ⏰ e.g., Monday_1_AM_to_2_AM_data.csv
│
├── data.py                 # 🧹 Data loading, preprocessing & feature engineering
│                           # - Loads from datasets/
│                           # - Cleans and formats for model input
│
├── model.py                # 🧠 Model building, training & saving
│                           # - Trains LSTM models
│                           # - Saves model (.keras) and scaler (.pkl) to TrainedModel/
│
├── predict.py              # 🔮 Forecast logic
│                           # - Loads saved models & scalers
│                           # - Predicts demand for a given time interval
│                           # - Used by both FastAPI and Streamlit UI
│
├── api.py                  # 🌐 FastAPI backend
│                           # - REST endpoints for prediction
│                           # - Uses predict.py for forecast logic
│
├── ui/
│   └── streamlit_app.py    # 🖼️ Streamlit frontend for:
│                           # - Model training
│                           # - Forecasting (15-min, hourly, etc.)
│                           # - Map visualization of predictions
│
├── Datasets_Map_html/      # 🗺️ Map visualizations (HTML files)
│   └── *.html              # e.g., Monday_1_AM_to_2_AM_data.html
│                           # - Used for display in Streamlit UI
│
├── TrainedModel/           # 💾 Saved ML artifacts
│   ├── *.keras             # ✅ Trained LSTM models
│   ├── *.pkl               # ✅ Scalers and encoders
│   └── forecast_outputs/   # 📊 Output CSVs with forecasted values
│       └── *.csv           # e.g., forecast_Monday_1_AM_to_2_AM.csv


## File & Folder Descriptions

### Top-Level Python Scripts
- **api.py**: FastAPI app for serving model predictions and training endpoints.
- **config.py**: Stores project-wide constants, file paths, and configuration settings.
- **data.py**: Handles loading, cleaning, and preprocessing of raw taxi data.
- **model.py**: Contains functions for building, training, and saving LSTM models.
- **predict.py**: Provides prediction logic; used by both the API and Streamlit UI.
- **requirements.txt**: Lists all Python dependencies needed for the project.
- **README.md**: Basic project introduction.
- **Project_Readme.md**: (This file) Full documentation of project structure and usage.

### Data Folders
- **datasets/**: Contains CSV files for each weekday and time slot, used for model training.
- **Datasets_Map_html/**: Contains HTML files for visualizing dataset samples on a map (for QA and presentation).
- **TrainedModel/**: Stores all trained model files, scalers, encoders, and forecast outputs.
    - **forecast_outputs/**: Contains forecasted passenger demand CSVs for each trained model.

### UI
- **ui/streamlit_app.py**: The main Streamlit app. Provides:
    - Model training interface
    - Demand forecasting (city-wide and by H3 hex)
    - Map visualizations of predictions

---

## Usage Overview

- **Train Model**: Use the Streamlit UI to select a time slot and train a new demand model. Trained models and scalers are saved in `TrainedModel/`.
- **Forecast Demand**: Use the UI to forecast demand for the entire city or a specific location (by H3 index). Forecast results are saved in `TrainedModel/forecast_outputs/`.
- **API Access**: Use `api.py` to serve predictions and model training via REST endpoints.

---

## Notes
- All time slots are handled as separate models and files (e.g., Monday_1_AM_to_2_AM).
- Map visualizations use H3 hexagons and Folium.
- Reinforcement learning logic for driver dispatch is in `rl.py` (optional).

For further details, see code comments in each file or open an issue.
