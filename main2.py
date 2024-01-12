import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

st.markdown(
    """<style>
        .title {
            background-color: #4CAF50
        }
    </style>""",

    unsafe_allow_html = True
)

model = pickle.load(open("model.pkl", "rb"))

st.title("Song Popularity")

#Taking Input
acousticness = st.text_input("Acousticness")
danceability = st.text_input("Danceability")
energy = st.text_input("Energy")
instrumentalness = st.text_input("Instrumentalness")
liveness = st.text_input("liveness")
loudness = st.text_input("loudness")
speechiness = st.text_input('speechiness')
tempo = st.text_input("Tempo")
audio_valence = st.text_input("Audio_valence")
song_duration_min = st.text_input("Song_duration_minutes")
audio_intensity = st.text_input("audio_intensity")
liveness_dance = st.text_input("liveness dance")
key = st.number_input("Key",min_value=0, max_value=11, step=1, value=0)
audio_mode = st.number_input("Audio Mode", min_value=0, max_value=1)
time_signature = st.text_input("time_signature")
instrumental = st.text_input("instrumental")

def predict(acousticness, danceability,energy, instrumentalness,
            liveness, loudness, speechiness,tempo,
            audio_valence, song_duration_min,  
            audio_intensity, liveness_dance,  key, 
            audio_mode, time_signature, instrumental):
    try:
        input_data ={
            'numerical__acousticness' : float(acousticness),
            'numerical__danceability' : float(danceability),
            'numerical__energy' : float(energy),
            "numerical__Instrumentalness" : float(instrumentalness),
            'numerical__liveness' : float(liveness),
            'numerical__loudness' : float(loudness),
            'numerical__speechiness' : float(speechiness),
            "numerical__tempo" : float(tempo),
            'numerical__audio_valence' : float(audio_valence),
            'numerical__song_duration_min' : float(song_duration_min),
            'numerical__audio_intensity' : float(audio_intensity),
            'numerical__liveness_dance' : float(liveness_dance),
            'categorical_pipeline__key' : float(key),
            'categorical_pipeline__audio_mode' : float(audio_mode),
            'categorical_pipeline__time_signature' : float(time_signature),
            'categorical_pipeline__instrumental' : float(instrumental)
        }

        data = pd.DataFrame([input_data])
        #Since we scaled the data we used to train the models, we should also scale the data to be predicted
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        pred = model.predict(scaled_data)[0]

        song_type = {0: "Not Popular", 1: "Popular"}
        song_pred = song_type.get(pred, "Unknown")
        st.success(f"The Song is {song_pred}")

    except ValueError:
        st.error("Please Enter A Valid Error")
if st.button("Predict"):
    predict(acousticness, danceability,energy, instrumentalness,
            liveness, loudness, speechiness,tempo,
            audio_valence, song_duration_min,  
            audio_intensity, liveness_dance,  key, 
            audio_mode, time_signature, instrumental)
    st.snow()
