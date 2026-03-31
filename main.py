from fastapi import FastAPI
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import requests

app = FastAPI()

# Path to model file
MODEL_PATH = "weather_model.keras"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    url = "https://your-storage-link/weather_model.keras"  # replace with Hugging Face / Google Drive / S3 link
    r = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load your model
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer="adam", loss="mse")

features = ["temperature","humidity","wind_speed","rainfall",
            "pressure","cloud_cover","solar_radiation",
            "dew_point","visibility","gas_resistance"]

def prepare_data():
    data = pd.read_csv("weather_data.csv")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    return data, scaler, scaled_data

@app.get("/forecast/7days")
def forecast_temperature():
    data, scaler, scaled_data = prepare_data()
    last_sequence = np.expand_dims(scaled_data[-30:], axis=0)
    prediction = model.predict(last_sequence)
    predicted_weather = scaler.inverse_transform(
        np.hstack([prediction.reshape(-1,1), np.zeros((7, len(features)-1))])
    )[:,0]
    return {"7_day_forecast_temperature": predicted_weather.tolist()}

@app.get("/aqi")
def forecast_aqi():
    data, _, _ = prepare_data()
    gas = data["gas_resistance"].tail(7).tolist()
    aqi = [max(0, 500 - g/10) for g in gas]  # placeholder formula
    return {"aqi_forecast": aqi}
