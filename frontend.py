import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import os
import requests  # For optional API call
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

# Load model (for direct prediction)
model_path = "best_floodnet_model.pth"
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} not found.")
    st.stop()
model = FloodNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# App title and description
st.title("Flood Risk Prediction")
st.markdown("Enter environmental features to predict flood risk using the FloodNet model.")

# Input form
with st.form(key="prediction_form"):
    st.subheader("Categorical Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vegetation = st.selectbox("Vegetation", options=valid_categories["Vegetation"])
        urbanization = st.selectbox("Urbanization", options=valid_categories["Urbanization"])
        drainage = st.selectbox("Drainage", options=valid_categories["Drainage"])
    with col2:
        slope = st.selectbox("Slope", options=valid_categories["Slope"])
        storm_frequency = st.selectbox("Storm Frequency", options=valid_categories["StormFrequency"])
        deforestation = st.selectbox("Deforestation", options=valid_categories["Deforestation"])
        
    with col3:
        infrastructure = st.selectbox("Infrastructure", options=valid_categories["Infrastructure"])
        encroachment = st.selectbox("Encroachment", options=valid_categories["Encroachment"])
        season = st.selectbox("Season", options=valid_categories["Season"])
    
    st.subheader("Other Features")
    col4, col5 = st.columns(2)
    
    with col4:
        soil = st.selectbox("Soil", options=valid_categories["Soil"])
        wetlands = st.selectbox("Wetlands", options=valid_categories["Wetlands"])
        proximity = st.selectbox("Proximity to Water Body", options=valid_categories["ProximityToWaterBody"])
    
    with col5:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0, step=0.1)
        elevation = st.number_input("Elevation (m)", min_value=0.0, value=0.0, step=0.1)
    
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

    # Option 1: Direct Model Prediction
    try:
        # Preprocess input (mirroring api.py)
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
        
        # Display results
        st.success(f"**Predicted Flood Risk**: {class_labels[pred_idx]}")
        st.subheader("Probability Breakdown")
        for label, prob in probs_dict.items():
            st.write(f"{label}: {prob}")
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

    # Option 2: API Call (uncomment to use FastAPI, comment out direct prediction above)
    """
    try:
        # Prepare payload for API
        payload = user_df.to_dict(orient="records")[0]
        response = requests.post("http://localhost:9000/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Display results
        st.success(f"**Predicted Flood Risk**: {result['predicted_class']}")
        st.subheader("Probability Breakdown")
        for label, prob in result['probabilities'].items():
            st.write(f"{label}: {prob}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
    """

# Footer
st.markdown("---")
st.write("Built with Streamlit and the FloodNet model from [opeblow/flood-risk-prediction-model](https://github.com/opeblow/flood-risk-prediction-model).")
