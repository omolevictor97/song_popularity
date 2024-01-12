from pydantic import BaseModel
import uvicorn
import requests
import json
from fastapi import FastAPI, File, Form, UploadFile
from io import StringIO, BytesIO
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

class SongPrediction(BaseModel):
    numerical__acousticness: float
    numerical__danceability: float
    numerical__energy : float
    numerical__instrumentalness: float
    numerical__liveness: float
    numerical__loudness: float
    numerical__speechiness: float
    numerical__tempo: float
    numerical__audio_valence: float
    numerical__song_duration_min: float
    numerical__audio_intensity: float
    numerical__liveness_dance: float
    categorical_pipeline__key: float
    categorical_pipeline__audio_mode: float
    categorical_pipeline__time_signature : float
    categorical_pipeline__instrumental : float

@app.get("/")
async def root():
    return {"Message": "MLFLOW FASTAPI TESTING"}

@app.post("/predict")
async def predict_song_popularity(song: SongPrediction):
    # Generate a dictionary representation of the pydantic BaseModel

    # Prepare the data in the most suitable format for Mlflow
    data_input = pd.DataFrame([dict(song)])
    print(data_input)

    # Define the mlflow model server endpoint
    endpoint = "http://localhost:5000/invocations"

    # Prepare the inference request payload
    inference_request = {"dataframe_records": data_input.to_dict(orient = "records")}
    print(inference_request)

    # Make the POST request to the MLflow model server and get the response
    response = requests.post(endpoint, json=inference_request)
    print(response.text)

    song_mapping = {0: "Song Not Popular", 1: "Song Popular"}

    # Extract prediction from response
    prediction = response.json().get("predictions")[0]
    popularity = song_mapping.get(prediction, "Unknown")

    return {"Prediction": popularity}

#Batch prediction
@app.post("/files")
async def batch_predict(file: bytes = File(...)):
    # Assuming the file is in CSV format
    s = str(file, encoding='UTF-8')
    data = StringIO(s)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(data)
    
    # Convert the DataFrame to a dictionary for the inference request
    inference_request = {"dataframe_records": df.to_dict(orient='records')}
    
    endpoint = 'http://localhost:5000/invocations'
    response = requests.post(endpoint, json=inference_request)
    
    song_pred_mapping = {0: "Not Popular", 1: "Popular"}
    predictions = response.json().get("predictions", [])
    
    # Map predictions to human-readable labels
    pred = [song_pred_mapping.get(pred, "unknown") for pred in predictions]
    
    return {"Predictions": pred}
