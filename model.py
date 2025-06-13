"""
model.py

🧠 Model Construction, Training, and Serialization for NYC Taxi Demand Forecasting
----------------------------------------------------------------------------------
This module is responsible for:
- Building an LSTM-based neural network that forecasts taxi demand using temporal and spatial features.
- Embedding H3 geospatial encodings to include spatial context in the LSTM model.
- Scaling input and output data using MinMaxScaler.
- Saving trained model and preprocessing components (scaler and label encoder) for future inference.

Functions:
- build_and_train_model: Constructs and trains an LSTM model that includes embedded H3 location data.
- save_model_components: Persists the trained Keras model and preprocessing tools to disk.

"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Flatten, RepeatVector
import tensorflow as tf
import joblib
import os
from typing import Tuple
from config import TRAINED_MODEL_DIR

# === MODEL BUILDING & TRAINING ===
def build_and_train_model(X: np.ndarray, y: np.ndarray, hex_ids: np.ndarray, agg_df, epochs: int = 40) -> Tuple[Model, MinMaxScaler]:
    """
    Build and train LSTM model with embedded H3 hex encoding.
    Args:
        X: LSTM input sequences.
        y: Target values.
        hex_ids: Encoded H3 hex IDs.
        agg_df: Aggregated DataFrame for embedding size.
        epochs: Number of training epochs.
    Returns:
        model: Trained Keras model.
        scaler: Fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()

    X_lstm = X_scaled[..., np.newaxis]
    hex_ids_input = hex_ids[..., np.newaxis]

    input_hex = Input(shape=(1,), name='hex_input')
    input_seq = Input(shape=(X.shape[1], 1), name='sequence_input')

    embedding = Embedding(input_dim=agg_df['h3_encoded'].nunique(), output_dim=8)(input_hex)
    embedding = Flatten()(embedding)
    embedding = RepeatVector(X.shape[1])(embedding)

    x = Concatenate()([input_seq, embedding])
    x = LSTM(32)(x)
    output = Dense(1, activation='relu')(x)

    model = Model(inputs=[input_hex, input_seq], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit([hex_ids_input, X_lstm], y_scaled, epochs=epochs, batch_size=32, validation_split=0.1)
    return model, scaler

# === SAVE MODEL COMPONENTS ===
def save_model_components(model: Model, scaler: MinMaxScaler, le: LabelEncoder, file_prefix: str):
    """
    Save trained model and preprocessing components.
    Args:
        model: Trained Keras model.
        scaler: Fitted MinMaxScaler.
        le: Fitted LabelEncoder.
        file_prefix: Prefix for saved files.
    """
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
    model.save(os.path.join(TRAINED_MODEL_DIR, f'{file_prefix}_model.keras'))
    joblib.dump(scaler, os.path.join(TRAINED_MODEL_DIR, f'{file_prefix}_scaler.pkl'))
    joblib.dump(le, os.path.join(TRAINED_MODEL_DIR, f'{file_prefix}_labelencoder.pkl'))
