import streamlit as st
import sounddevice as sd
import numpy as np
import torch
from transformers import pipeline
import time

# Load Whisper model from Hugging Face
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Configure audio settings
SAMPLERATE = 16000  # Whisper expects 16kHz sample rate
CHANNELS = 1


# Function to capture and transcribe audio
def transcribe_audio(indata):
    # Convert the audio data to numpy array (if it's a torch tensor)
    audio_np = indata[:, 0].astype(np.float32)

    # Whisper expects numpy array, so we pass it directly
    transcription = whisper_model(audio_np)
    return transcription["text"]


# Streamlit UI setup
st.title("Voice Transcription App")
st.write("Record your voice, and the app will transcribe it using Whisper.")

record_button = st.button("Start Recording")  # record button

transcription_text = st.empty()  # display transcription

if record_button:
    with st.spinner("Recording..."):
        audio_data = sd.rec(
            int(SAMPLERATE * 5),
            samplerate=SAMPLERATE,
            channels=CHANNELS,  # record for 5 seconds
        )
        sd.wait()  # wait for recording to finish

        transcription = transcribe_audio(audio_data)  # transcribe audio

        transcription_text.write(f"**Transcription:** {transcription}")

        time.sleep(1)  # pause for ui transition
