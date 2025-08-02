# 🌊 Flood Risk Prediction Model 🌧️

## 📝 Overview
Welcome to the **Flood Risk Prediction Model**! 🚨 This project uses machine learning to predict flood risk based on environmental features like elevation, rainfall, soil type, and proximity to water bodies. It includes a training script (`cmodel_1.py`), a **FastAPI** backend (`api.py`), and a **Streamlit** frontend (`frontend.py`) with an interactive map 🗺️ for a user-friendly experience. Whether you're a researcher, developer, or curious user, this tool helps assess flood risk with ease! 😄

## 🎯 Features
- **Interactive Map** 🗺️: Click on a world map in the Streamlit app to select a location (latitude/longitude).
- **Auto-Fill Data** 🌍: Automatically fetches data for features like:
  - **Elevation** 📏: From Open-Meteo API.
  - **Rainfall** ☔: Sum of daily rain over the past 7 days (Open-Meteo).
  - **Soil Type** 🌱: From SoilGrids API (classifies as Clay, Sandy, Loamy, or Rocky).
  - **Proximity to Water Body** 🏞️: Uses Overpass API (OSM) to find the nearest river, stream, lake, lagoon, or dam and classifies distance (e.g., "Very Close to River").
  - **Wetlands** 🌿: Checks for wetlands within 1km (Overpass API).
  - **Slope** ⛰️: Approximates slope by comparing elevations at nearby points.
  - **Season** 🌞: Auto-set based on current month and hemisphere (Rainy, Dry, Transition).
- **Manual Inputs** ✍️: Selectboxes for features like Vegetation, Urbanization, etc., with sensible defaults (e.g., "Moderate").
- **Prediction** 🔍: Submits data to the FloodNet model (or API) to predict flood risk (None, Low, Medium, High).
- **Map Visualization** 🎨: Marks the clicked location with a color-coded marker (🟥 High, 🟧 Medium, 🟩 Low, ⬜ None).
- **API Backend** ⚙️: FastAPI serves predictions and valid category options.
- **No API Keys Needed** 🔓: Uses free, public APIs for data fetching.

## 🛠️ Setup
Follow these steps to get started! 🚀

1. **Clone the Repository** 📂:
   ```bash
   git clone https://github.com/opeblow/flood-risk-prediction-model.git
   cd flood-risk-prediction-model
   ```

2. **Install Dependencies** 📦:
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages:
   - `torch` 🧠
   - `pandas` 📊
   - `scikit-learn` 📈
   - `streamlit` 🌐
   - `fastapi` ⚙️
   - `uvicorn` 🖥️
   - `folium` 🗺️
   - `streamlit-folium` 🗺️
   - `requests` 🌐
   - `haversine` 📍

3. **Prepare Data and Model** 📁:
   - Ensure `flood_risk_dataset_final.csv` and `best_floodnet_model.pth` are in the project root.
   - The dataset contains environmental data; the model is a pre-trained FloodNet neural network.

4. **Run the Application** 🚀:
   Use the provided `start.sh` script to launch both the FastAPI backend and Streamlit frontend:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
   - **FastAPI**: Runs on `http://localhost:9000`
   - **Streamlit**: Runs on `http://localhost:8501`

## 📚 Usage
1. **Open the Streamlit App** 🌐:
   Navigate to `http://localhost:8501` in your browser.

2. **Interact with the Map** 🗺️:
   - Click a location on the world map to capture latitude and longitude.
   - The app fetches environmental data for the selected location (e.g., elevation, rainfall).

3. **Review Auto-Filled Form** ✍️:
   - The form auto-updates with fetched values (e.g., Soil, Wetlands, Proximity).
   - Manually adjust other features (e.g., Vegetation, Urbanization) via selectboxes.

4. **Predict Flood Risk** 🔍:
   - Click "Predict Flood Risk" to submit the form.
   - The app processes inputs using the FloodNet model and displays the predicted risk (None, Low, Medium, High) with probability breakdown.

5. **Visualize on Map** 🎨:
   - The clicked location is marked with a color-coded marker based on the predicted risk.

6. **API Access** ⚙️:
   - Access the FastAPI backend at `http://localhost:9000`.
   - Endpoints:
     - `POST /predict`: Send feature data to get flood risk predictions.
     - `GET /valid_options`: Retrieve valid category options for inputs.

## 📝 Notes
- **API Details** 🌐:
  - **Open-Meteo**: Provides elevation and recent rainfall (7-day sum). Free, 10k calls/day limit.
  - **SoilGrids**: Fetches soil composition (clay, sand, silt) at 0-5cm depth. Classified into model-compatible types (Clay, Sandy, Loamy, Rocky).
  - **Overpass API (OSM)**: Finds nearby water bodies (within 5km) and wetlands (within 1km). Distance-based classification for ProximityToWaterBody.
  - **Season**: Determined by current month and hemisphere (simplified for tropics; adjust for specific climates).
  - **Slope**: Approximated using elevation differences over ~100m offsets. May need refinement for accuracy.

- **Limitations** ⚠️:
  - Rainfall is based on recent 7-day data; for annual/historical rainfall, consider APIs like World Bank Climate.
  - Slope calculation is approximate; more precise DEM (Digital Elevation Model) APIs could improve accuracy.
  - Features like Vegetation, Urbanization, etc., use manual inputs with defaults. Auto-filling these requires advanced land cover APIs (e.g., Google Earth Engine, paid).
  - API rate limits apply (e.g., Open-Meteo’s 10k calls/day). Cache results for production use.
  - Map markers persist in session state; clear manually if needed.

- **Potential Improvements** 🚀:
  - Integrate paid APIs (e.g., Earth Engine) for Vegetation, Urbanization, etc.
  - Use historical/seasonal rainfall data for better accuracy.
  - Enhance slope calculation with high-resolution DEM data.
  - Add caching for API calls to handle rate limits.
  - Allow users to save predictions or export map visualizations.
  - Deploy to cloud platforms (e.g., Heroku, AWS) for public access.

- **Dependencies** 📦:
  Ensure all required packages are installed. Update `requirements.txt` with specific versions for reproducibility.

- **Model Details** 🧠:
  - The `FloodNet` model is a neural network trained on `flood_risk_dataset_final.csv`.
  - Features are preprocessed (e.g., numerical clipping, log transformation, encoding) to match training conditions.
  - Predictions are based on a softmax output, providing probabilities for each class (None, Low, Medium, High).

## 🙌 Contributing
Contributions are welcome! 😊 Fork the repo, make improvements, and submit a pull request. Ideas:
- Enhance API integrations 🌐
- Improve UI/UX in Streamlit 🎨
- Add more robust data validation ✅
- Optimize model performance ⚡

## 📬 Contact
For questions, reach out via [GitHub Issues](https://github.com/opeblow/flood-risk-prediction-model) or contact [raimonvibe](https://about.me/raimonvibe). 📧

## 🔗 Credits
Built with ❤️ using [opeblow/flood-risk-prediction-model](https://github.com/opeblow/flood-risk-prediction-model). Enhanced with map integration and API auto-fill by Grok, created by xAI. 🚀
