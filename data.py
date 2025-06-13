"""
data.py

📊 Data Preprocessing & Sequence Preparation for NYC Taxi Demand Forecasting
----------------------------------------------------------------------------
This module handles:
- Loading and filtering raw CSV data based on weekday and time interval.
- Aggregating passenger count data by week, hour, and H3 hex index.
- Encoding spatial features using LabelEncoder.
- Creating sequences of historical data suitable for training LSTM models.

Functions:
- preprocess_data: Reads and filters input data, aggregates by time and location, encodes spatial index.
- create_sequences: Converts the aggregated data into weekly input-output sequences for model training.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional

# === DATA PREPROCESSING ===
def preprocess_data(file_path: str, start_hour: int, end_hour: int, weekday_num: int) -> Tuple[Optional[pd.DataFrame], Optional[LabelEncoder]]:
    """
    Preprocess input CSV and return aggregated DataFrame and LabelEncoder.
    Args:
        file_path: Path to the CSV data file.
        start_hour: Start hour in 24-hour format.
        end_hour: End hour in 24-hour format.
        weekday_num: 0=Monday, 6=Sunday.
    Returns:
        agg_df: Aggregated DataFrame by week, hour, and h3_index.
        le: Fitted LabelEncoder for h3_index.
    """
    df = pd.read_csv(file_path)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['week'] = df['tpep_pickup_datetime'].dt.isocalendar().week
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df = df[df['day_of_week'] == weekday_num]
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df = df[(df['pickup_hour'] >= start_hour) & (df['pickup_hour'] < end_hour)]

    if df.empty:
        return None, None

    agg_df = df.groupby(['week', 'pickup_hour', 'h3_index'])['passenger_count'].sum().reset_index()
    if agg_df.empty:
        return None, None

    le = LabelEncoder()
    agg_df['h3_encoded'] = le.fit_transform(agg_df['h3_index'])

    return agg_df, le

# === LSTM SEQUENCE CREATION ===
def create_sequences(df: pd.DataFrame, n_weeks: int = 8):
    """
    Create training sequences of passenger counts by h3_id and hour.
    Args:
        df: Aggregated DataFrame.
        n_weeks: Number of weeks in sequence.
    Returns:
        sequences, targets, hex_ids, hours: Arrays for LSTM input.
    """
    sequences, targets, hex_ids, hours = [], [], [], []
    if df is None or df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])
    for (h3_id, hour), group in df.groupby(['h3_encoded', 'pickup_hour']):
        group = group.sort_values('week')
        values = group['passenger_count'].values
        if len(values) >= n_weeks + 1:
            for i in range(n_weeks, len(values)):
                sequences.append(values[i - n_weeks:i])
                targets.append(values[i])
                hex_ids.append(h3_id)
                hours.append(hour)
    return np.array(sequences), np.array(targets), np.array(hex_ids), np.array(hours)
