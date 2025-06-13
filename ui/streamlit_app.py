"""
streamlit_app.py
Streamlit UI for NYC Taxi Demand Forecast project.
"""
import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import random
from config import TRAINED_MODEL_DIR, OUTPUT_FOLDER, WEEKDAYS
from predict import predict_demand

# === PAGE SELECTOR ===
st.set_page_config(page_title="NYC Taxi Demand Forecast", layout="wide")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🚀 Train New Model", "🔍 Forecast Demand"])

# === PAGE 1: TRAIN MODEL ===
def train_model_page():
    st.title("🚖 NYC Taxi Demand Forecast - Train Model")
    st.subheader("Train New Demand Forecast Model")

    weekday = st.selectbox("Select Weekday", list(range(7)), format_func=lambda x: WEEKDAYS[x])
    start_hour = st.selectbox("Start Hour", list(range(1, 13)))
    am_pm_start = st.selectbox("Start Meridiem", ["AM", "PM"])
    end_hour = st.selectbox("End Hour", list(range(1, 13)))
    am_pm_end = st.selectbox("End Meridiem", ["AM", "PM"])

    weekday_str = WEEKDAYS[weekday]
    start_hour_24 = 0 if start_hour == 12 else start_hour
    start_hour_24 += 12 if am_pm_start == "PM" and start_hour != 12 else 0
    end_hour_24 = 0 if end_hour == 12 else end_hour
    end_hour_24 += 12 if am_pm_end == "PM" and end_hour != 12 else 0

    dataset_filename = os.path.join("datasets", f"{weekday_str}_{start_hour}_{am_pm_start}_to_{end_hour}_{am_pm_end}_data.csv")

    if os.path.exists(dataset_filename):
        st.info(f"✅ Dataset found: `{dataset_filename}`")
    else:
        st.warning(f"⚠️ Dataset not found: `{dataset_filename}`")

    if st.button("🚀 Train Model"):
        if os.path.exists(dataset_filename):
            with st.spinner("Training model..."):
                response = requests.post("http://localhost:8000/train_model/", json={
                    "weekday_num": weekday,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "am_pm_start": am_pm_start,
                    "am_pm_end": am_pm_end
                })
                result = response.json()
                if "message" in result:
                    st.success("✅ Model trained and saved!")
                else:
                    st.error(result.get("error", "❌ Training failed."))
        else:
            st.error(f"❌ Cannot train model. Dataset not found: `{dataset_filename}`")

# === PAGE 2: FORECAST DEMAND ===
def forecast_demand_page():
    st.title("🚖 NYC Taxi Demand Forecasting")
    st.subheader("🔍 Predict demand using historical models by time slot and location")
    st.markdown("### 🗓️ Select Weekday")
    weekday = st.selectbox("Choose Weekday (0=Monday, ..., 6=Sunday)", list(range(7)), format_func=lambda x: WEEKDAYS[x])
    st.markdown("### ⏰ Forecast Specific Hour Interval")
    forecast_input = st.selectbox("Select Time Slot (1-12 for Hour)", list(range(1, 13)))
    am_pm = st.selectbox("Select AM/PM", ["AM", "PM"])

    col1, col2 = st.columns(2)
    with col1:
        forecast_entire_city = st.button("Forecast demand on entire city", key="entire_city_btn")
    with col2:
        forecast_specific_location = st.button("Forecast demand on specific location", key="specific_location_btn")

    # Build model prefix as per training
    weekday_name = WEEKDAYS[weekday]
    start_hour_24 = 0 if forecast_input == 12 else forecast_input
    start_hour_24 += 12 if am_pm == "PM" and forecast_input != 12 else 0
    end_hour_24 = (start_hour_24 + 1) % 24
    end_meridiem = "AM" if end_hour_24 < 12 or end_hour_24 == 24 else "PM"
    model_prefix = f"{weekday_name}_{forecast_input}_{am_pm}_to_{end_hour_24}_{end_meridiem}"

    # --- SESSION STATE LOGIC ---
    if "show_specific_location" not in st.session_state:
        st.session_state.show_specific_location = False

    if forecast_specific_location:
        st.session_state.show_specific_location = True
    if forecast_entire_city:
        st.session_state.show_specific_location = False

    if forecast_entire_city:
        # (existing entire city code unchanged)
        try:
            from predict import predict_demand
            with st.spinner(f"Loading model and forecasting for {model_prefix}..."):
                predictions = predict_demand(model_prefix)
            if predictions:
                st.success(f"✅ Forecast for {model_prefix} loaded.")
                df = pd.DataFrame(list(predictions.items()), columns=["h3_hex_id", "predicted_passenger_count"])
                df["predicted_passenger_count"] = df["predicted_passenger_count"].round().astype(int)
                output_dir = OUTPUT_FOLDER
                os.makedirs(output_dir, exist_ok=True)
                csv_path = os.path.join(output_dir, f"forecast_{model_prefix}.csv")
                df.to_csv(csv_path, index=False)
                st.success("📁 Forecast data saved")
                try:
                    import h3
                    import folium
                    from streamlit_folium import folium_static
                    from branca.colormap import linear

                    valid_rows = []
                    for _, row in df.iterrows():
                        h3_index = row["h3_hex_id"]
                        if h3.h3_is_valid(h3_index):
                            valid_rows.append(row)
                    if not valid_rows:
                        st.warning("No valid H3 hexagons found in predictions.")
                    else:
                        valid_df = pd.DataFrame(valid_rows)
                        nyc_center = [40.730610, -73.935242]
                        m = folium.Map(location=nyc_center, zoom_start=11, tiles="cartodbpositron")
                        color_scale = linear.YlOrRd_09.scale(valid_df["predicted_passenger_count"].min(), valid_df["predicted_passenger_count"].max())
                        for _, row in valid_df.iterrows():
                            h3_index = row["h3_hex_id"]
                            demand = row["predicted_passenger_count"]
                            try:
                                boundary = h3.h3_to_geo_boundary(h3_index)
                                boundary = [[lat, lng] for lat, lng in boundary]
                                popup_content = f"<b>H3 Index:</b> {h3_index}<br><b>Predicted Passengers:</b> {int(round(demand))}"
                                popup = folium.Popup(popup_content, max_width=400, min_width=300)
                                folium.Polygon(
                                    locations=boundary,
                                    color='black',
                                    weight=1,
                                    fill=True,
                                    fill_opacity=0.7,
                                    fill_color=color_scale(demand),
                                    popup=popup
                                ).add_to(m)
                            except Exception as e:
                                st.write(f"Error with hex {h3_index}: {e}")
                                continue
                        color_scale.caption = f"Predicted Passengers: {model_prefix}"
                        color_scale.add_to(m)
                        st.markdown("### 🗺️ Map of Predicted Passengers (H3 Hexagons)")
                        folium_static(m)
                except ImportError:
                    st.warning("Install h3, folium, and streamlit_folium for map visualization.")
            else:
                st.warning(f"No predictions found for {model_prefix}.")
        except Exception as e:
            st.error(f"❌ Could not forecast for {model_prefix}: {e}")

    elif st.session_state.show_specific_location:
        # Load the CSV first
        output_dir = OUTPUT_FOLDER
        model_csv = os.path.join(output_dir, f"forecast_{model_prefix}.csv")
        if os.path.exists(model_csv):
            df = pd.read_csv(model_csv)
            st.success(f"✅ Forecast for {model_prefix} loaded.")
            h3_input = st.text_input("Enter H3 Index ID (e.g., 882a100d67fffff):")
            if h3_input:
                match = df[df["h3_hex_id"] == h3_input]
                if not match.empty:
                    demand = int(match["predicted_passenger_count"].iloc[0])
                    try:
                        import h3
                        import folium
                        from streamlit_folium import folium_static
                        nyc_center = [40.730610, -73.935242]
                        m = folium.Map(location=nyc_center, zoom_start=12, tiles="cartodbpositron")
                        if h3.h3_is_valid(h3_input):
                            boundary = h3.h3_to_geo_boundary(h3_input)
                            boundary = [[lat, lng] for lat, lng in boundary]
                            popup_content = f"<b>H3 Index:</b> {h3_input}<br><b>Predicted Passengers:</b> {demand}"
                            popup = folium.Popup(popup_content, max_width=400, min_width=300)
                            folium.Polygon(
                                locations=boundary,
                                color='black',
                                weight=2,
                                fill=True,
                                fill_opacity=0.8,
                                fill_color='orange',
                                popup=popup
                            ).add_to(m)
                            lat_center, lng_center = h3.h3_to_geo(h3_input)
                            m.location = [lat_center, lng_center]
                            m.zoom_start = 13
                            st.markdown("### 🗺️ Map of Predicted Passengers (Selected H3 Hexagon)")
                            folium_static(m)
                        else:
                            st.error("Invalid H3 index.")
                    except ImportError:
                        st.warning("Install h3, folium, and streamlit_folium for map visualization.")
                else:
                    st.warning(f"No prediction found for H3 index {h3_input} in loaded forecast data.")
            else:
                st.info("Enter an H3 index ID above to forecast demand for a specific location.")
        else:
            st.warning(f"Forecast file not found: {model_csv}")
    else:
        st.info("Select a time slot and choose a forecast option above.")

# === ROUTING ===
if page == "🚀 Train New Model":
    train_model_page()
elif page == "🔍 Forecast Demand":
    forecast_demand_page()