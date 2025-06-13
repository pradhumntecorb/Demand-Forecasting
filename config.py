"""
config.py

🛠️ Configuration Module for NYC Taxi Demand Forecasting
-------------------------------------------------------
This module centralizes all configuration parameters and path settings
used across the project. It includes environment-based overrides for 
dataset, model, and output directories, and defines general constants 
like weekday mappings.

Configurations:
- Dataset folder path
- Trained model save directory
- Forecast output directory
- Weekday name mappings

"""

import os

# === DATASET & MODEL PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.getenv("DATASET_FOLDER", os.path.join(BASE_DIR, "datasets"))
TRAINED_MODEL_DIR = os.getenv("TRAINED_MODEL_DIR", os.path.join(BASE_DIR, "TrainedModel"))
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", os.path.join(TRAINED_MODEL_DIR, "forecast_outputs"))

# === GENERAL PARAMETERS ===
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

