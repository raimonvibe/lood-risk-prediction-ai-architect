import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import os
import requests
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import st_folium

from cmodel_1 import FloodNet, valid_categories  # Import model and valid categories

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature groups (from cmodel_1.py)
text_feature = "ProximityToWaterBody"
numerical_features = ["Elevation", "Rainfall"]
ordinal_features = [
    "Vegetation", "Urbanization", "Drainage", "Slope", "StormFrequency",
    "Deforestation", "Infrastructure", "Encroachment", "Season"
]
onehot_features = ["Soil", "Wetlands"]
class_labels = ["None", "Low", "Medium", "High"]

# Load dataset for fitting encoders and scaler
csv_path = "flood_risk_dataset_final.csv"
if not os.path.exists(csv_path):
    st.error(f"Dataset file {csv_path} not found.")
    st.stop()
df = pd.read_csv(csv_path)

# Fit encoders and scaler
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
text_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scaler = StandardScaler()

# Fill NA
for col in ordinal_features + onehot_features + [text_feature]:
    df[col] = df[col].fillna("Missing")

ord_enc.fit(df[ordinal_features])
text_enc.fit(df[[text_feature]])
onehot_enc.fit(df[onehot_features])
scaler.fit(df[numerical_features])

# Load model
model_path = "best_floodnet_model.pth"
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} not found.")
    st.stop()
model = FloodNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Helper functions for auto-fill
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def fetch_elevation(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['elevation'][0]
    return 0.0  # Default on error

def fetch_rainfall(lat, lon):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=rain_sum&timezone=auto&start_date={start_date}&end_date={end_date}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return sum(data['daily']['rain_sum'])  # Sum over 7 days
    return 0.0

def fetch_soil(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay&property=sand&property=silt&depth=0-5cm&value=mean"
    response = requests.get(url)
    if response.status_code == 200:
        layers = response.json()['properties']['layers']
        clay = next((l['depths'][0]['values']['mean'] for l in layers if l['name'] == 'clay'), 0) / 10  # To %
        sand = next((l['depths'][0]['values']['mean'] for l in layers if l['name'] == 'sand'), 0) / 10
        silt = next((l['depths'][0]['values']['mean'] for l in layers if l['name'] == 'silt'), 0) / 10
        if sand > 60:
            return "Sandy"
        elif clay > 40:
            return "Clay"
        elif silt + clay > 50:
            return "Loamy"
        return "Rocky"
    return "Loamy"  # Default

def fetch_proximity_and_wetlands(lat, lon):
    overpass_url = "https://overpass-api.de/api/interpreter"
    water_types = {
        'River': 'waterway=river',
        'Stream': 'waterway=stream',
        'Lake': 'natural=water and water=lake',
        'Lagoon': 'natural=water and water=lagoon',
        'Dam': 'waterway=dam or man_made=dam'
    }
    min_dist = float('inf')
    closest_type = ""
    for w_type, query in water_types.items():
        overpass_query = f"""
        [out:json];
        (way[{query}](around:5000,{lat},{lon});
         rel[{query}](around:5000,{lat},{lon}););
        out geom;
        """
        response = requests.post(overpass_url, data={'data': overpass_query})
        if response.status_code == 200:
            data = response.json()
            for elem in data['elements']:
                if 'geometry' in elem:
                    for point in elem['geometry']:
                        dist = haversine_distance(lat, lon, point['lat'], point['lon'])
                        if dist < min_dist:
                            min_dist = dist
                            closest_type = w_type

    if min_dist == float('inf'):
        proximity = "Very Far from River"
    else:
        if min_dist < 100:
            dist_class = "Very Close to"
        elif min_dist < 500:
            dist_class = "Close to"
        elif min_dist < 1000:
            dist_class = "Moderately Close to"
        elif min_dist < 3000:
            dist_class = "Far from"
        else:
            dist_class = "Very Far from"
        proximity = f"{dist_class} {closest_type}"

    # Wetlands
    wetland_query = f"""
    [out:json];
    (way["natural"="wetland"](around:1000,{lat},{lon});
     rel["natural"="wetland"](around:1000,{lat},{lon}););
    out geom;
    """
    response = requests.post(overpass_url, data={'data': wetland_query})
    wetlands = "Present" if response.status_code == 200 and response.json()['elements'] else "Absent"

    return proximity, wetlands

def fetch_slope(lat, lon):
    offsets = 0.001  # ~100m
    points = [
        (lat, lon),  # center
        (lat + offsets, lon),  # North
        (lat - offsets, lon),  # South
        (lat, lon + offsets),  # East
        (lat, lon - offsets)   # West
    ]
    elevations = [fetch_elevation(p[0], p[1]) for p in points]
    if any(e == 0 for e in elevations): return "Flat"  # Error fallback

    dist = haversine_distance(lat, lon, lat + offsets, lon)  # Vertical/horizontal dist
    slopes = []
    for i in [1,2,3,4]:
        rise = elevations[i] - elevations[0]
        slope_deg = atan2(rise, dist) * (180 / np.pi)
        slopes.append(abs(slope_deg))
    max_slope = max(slopes)
    if max_slope < 2:
        return "Flat"
    elif max_slope < 10:
        return "Moderate"
    return "Steep"

def get_season(lat):
    month = datetime.now().month
    if lat > 0:  # Northern
        if 6 <= month <= 9:
            return "Rainy"
        elif 3 <= month <= 5 or 10 <= month <= 11:
            return "Transition"
        return "Dry"
    else:  # Southern
        if 12 <= month <= 2 or month == 12:
            return "Rainy"
        elif 3 <= month <= 5 or 9 <= month <= 11:
            return "Transition"
        return "Dry"

# App title
st.title("Flood Risk Prediction with Map")

# Session state for fetched data and map markers
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'markers' not in st.session_state:
    st.session_state.markers = []

# Display interactive map
m = folium.Map(location=[0, 0], zoom_start=2)
m.add_child(folium.ClickForMarker())
map_data = st_folium(m, width=700, height=500, key="map")

if map_data and map_data.get('last_clicked'):
    clicked = map_data['last_clicked']
    st.session_state.lat = clicked['lat']
    st.session_state.lon = clicked['lng']
    st.write(f"Clicked location: Lat {st.session_state.lat:.4f}, Lon {st.session_state.lon:.4f}")

    # Fetch data
    try:
        elevation = fetch_elevation(st.session_state.lat, st.session_state.lon)
        rainfall = fetch_rainfall(st.session_state.lat, st.session_state.lon)
        soil = fetch_soil(st.session_state.lat, st.session_state.lon)
        proximity, wetlands = fetch_proximity_and_wetlands(st.session_state.lat, st.session_state.lon)
        slope = fetch_slope(st.session_state.lat, st.session_state.lon)
        season = get_season(st.session_state.lat)

        st.session_state.fetched_data = {
            "Elevation": elevation,
            "Rainfall": rainfall,
            "Soil": soil,
            "Wetlands": wetlands,
            "ProximityToWaterBody": proximity,
            "Slope": slope,
            "Season": season
        }
        st.success("Auto-filled data from APIs!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Input form with auto-filled defaults
with st.form(key="prediction_form"):
    st.subheader("Categorical Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vegetation = st.selectbox("Vegetation", options=valid_categories["Vegetation"], index=valid_categories["Vegetation"].index("Moderate"))
        urbanization = st.selectbox("Urbanization", options=valid_categories["Urbanization"], index=valid_categories["Urbanization"].index("Medium"))
        drainage = st.selectbox("Drainage", options=valid_categories["Drainage"], index=valid_categories["Drainage"].index("Moderate"))
    with col2:
        slope_default = st.session_state.fetched_data.get("Slope", "Moderate")
        slope = st.selectbox("Slope", options=valid_categories["Slope"], index=valid_categories["Slope"].index(slope_default))
        storm_frequency = st.selectbox("Storm Frequency", options=valid_categories["StormFrequency"], index=valid_categories["StormFrequency"].index("Occasional"))
        deforestation = st.selectbox("Deforestation", options=valid_categories["Deforestation"], index=valid_categories["Deforestation"].index("Moderate"))
        
    with col3:
        infrastructure = st.selectbox("Infrastructure", options=valid_categories["Infrastructure"], index=valid_categories["Infrastructure"].index("Moderate"))
        encroachment = st.selectbox("Encroachment", options=valid_categories["Encroachment"], index=valid_categories["Encroachment"].index("Moderate"))
        season_default = st.session_state.fetched_data.get("Season", "Transition")
        season = st.selectbox("Season", options=valid_categories["Season"], index=valid_categories["Season"].index(season_default))
    
    st.subheader("Other Features")
    col4, col5 = st.columns(2)
    
    with col4:
        soil_default = st.session_state.fetched_data.get("Soil", "Loamy")
        soil = st.selectbox("Soil", options=valid_categories["Soil"], index=valid_categories["Soil"].index(soil_default))
        wetlands_default = st.session_state.fetched_data.get("Wetlands", "Absent")
        wetlands = st.selectbox("Wetlands", options=valid_categories["Wetlands"], index=valid_categories["Wetlands"].index(wetlands_default))
        proximity_default = st.session_state.fetched_data.get("ProximityToWaterBody", "Far from River")
        proximity = st.selectbox("Proximity to Water Body", options=valid_categories["ProximityToWaterBody"], index=valid_categories["ProximityToWaterBody"].index(proximity_default))
    
    with col5:
        rainfall_default = st.session_state.fetched_data.get("Rainfall", 0.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=rainfall_default, step=0.1)
        elevation_default = st.session_state.fetched_data.get("Elevation", 0.0)
        elevation = st.number_input("Elevation (m)", min_value=0.0, value=elevation_default, step=0.1)
    
    submit = st.form_submit_button("Predict Flood Risk")

# Prediction logic
if submit:
    # Prepare input DataFrame
    input_data = {
        "Vegetation": [vegetation],
        "Urbanization": [urbanization],
        "Drainage": [drainage],
        "Slope": [slope],
        "StormFrequency": [storm_frequency],
        "Deforestation": [deforestation],
        "Infrastructure": [infrastructure],
        "Encroachment": [encroachment],
        "Season": [season],
        "Soil": [soil],
        "Wetlands": [wetlands],
        "ProximityToWaterBody": [proximity],
        "Rainfall": [rainfall],
        "Elevation": [elevation]
    }
    user_df = pd.DataFrame(input_data)

    try:
        # Preprocess (same as original)
        for col in ordinal_features + onehot_features + [text_feature]:
            user_df[col] = user_df[col].fillna("Missing")
        
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
        
        # Predict
        with torch.no_grad():
            output = model(text_tensor, ordinal_tensor, onehot_tensor, num_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            probs_dict = {label: f"{round(float(probs[0][i])*100, 2)}%" for i, label in enumerate(class_labels)}
        
        predicted_risk = class_labels[pred_idx]
        st.session_state.prediction = predicted_risk
        
        # Display results
        st.success(f"**Predicted Flood Risk**: {predicted_risk}")
        st.subheader("Probability Breakdown")
        for label, prob in probs_dict.items():
            st.write(f"{label}: {prob}")
        
        # Add marker to map
        color_map = {"High": "red", "Medium": "orange", "Low": "green", "None": "white"}
        st.session_state.markers.append((st.session_state.lat, st.session_state.lon, color_map.get(predicted_risk, "blue")))
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Redisplay map with markers
if st.session_state.markers:
    m = folium.Map(location=[0, 0], zoom_start=2)
    for lat, lon, color in st.session_state.markers:
        folium.Marker([lat, lon], icon=folium.Icon(color=color)).add_to(m)
    st_folium(m, width=700, height=500, key="updated_map")

# Footer
st.markdown("---")
st.write("Built with Streamlit and the FloodNet model from [opeblow/flood-risk-prediction-model](https://github.com/opeblow/flood-risk-prediction-model). Enhanced with map and API auto-fill.")
