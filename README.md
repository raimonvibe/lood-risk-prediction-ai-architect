# Flood Risk Prediction Model

## Overview
A machine learning model for predicting flood risk based on environmental features. Includes training script, FastAPI backend, and Streamlit frontend with interactive map.

## Setup
1. Install dependencies: `pip install -r requirements.txt` (add torch, pandas, scikit-learn, streamlit, fastapi, uvicorn, folium, streamlit-folium, requests, haversine).
2. Run: `./start.sh`

## Features
- Interactive map for location selection.
- Auto-fills features using free APIs (Open-Meteo, SoilGrids, Overpass/OSM).
- Predicts flood risk and marks map with color-coded risk.

See code for details.
