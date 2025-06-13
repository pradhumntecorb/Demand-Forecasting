# Project_Readme

## NYC Taxi Demand Forecasting System — Project Overview

This project provides a modular, production-ready, and scalable pipeline for forecasting NYC taxi demand using LSTM-based time series models, with optional reinforcement learning for driver dispatch. It includes a REST API (FastAPI) and an interactive UI (Streamlit).

---

## Project Structure

project_root/
│
├── config.py               # ✅ Central config: paths, constants, global settings
├── requirements.txt        # ✅ Project dependencies (FastAPI, Streamlit, Keras, etc.)
├── Project_Readme.md       # ✅ Complete project documentation (detailed instructions)
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
