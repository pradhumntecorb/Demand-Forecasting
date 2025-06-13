"""
predict.py
Prediction logic for NYC Taxi Demand Forecast project.
"""
import os
import numpy as np
import joblib
import tensorflow as tf
from typing import Dict
from config import TRAINED_MODEL_DIR

def predict_demand(model_prefix: str) -> Dict[str, float]:
    """
    Load model and predict demand for all hexes for given time.
    Args:
        model_prefix: Prefix for saved model/scaler/encoder files.
    Returns:
        Dictionary mapping h3_hex_id to predicted passenger count.
    """
    model_path = os.path.join(TRAINED_MODEL_DIR, f'{model_prefix}_model.keras')
    scaler_path = os.path.join(TRAINED_MODEL_DIR, f'{model_prefix}_scaler.pkl')
    le_path = os.path.join(TRAINED_MODEL_DIR, f'{model_prefix}_labelencoder.pkl')

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
        raise FileNotFoundError(f"Model or preprocessing files not found for prefix: {model_prefix}")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    all_h3_indices = le.classes_
    encoded_h3_indices = le.transform(all_h3_indices)
    dummy_sequence = np.zeros((len(encoded_h3_indices), 8, 1))

    predictions_scaled = model.predict([encoded_h3_indices, dummy_sequence])
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    return dict(zip(all_h3_indices, predictions))
