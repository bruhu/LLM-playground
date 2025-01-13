import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import pipeline

# Load Whisper model from Hugging Face
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Configure audio settings
SAMPLERATE = 16000  # Whisper expects 16kHz sample rate
CHANNELS = 1

# Initialize session state for controlling the recording process
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None  # Initialize audio data


# Function to capture and transcribe audio
def transcribe_audio(indata):
    """Transcribe audio data using the Whisper model."""
    audio_np = indata[:, 0].astype(np.float32)  # Convert to float32
    transcription = whisper_model(audio_np)  # Transcribe audio
    return transcription["text"]


# Streamlit UI setup
st.title("Voice Transcription App")
st.write("Record your voice, and the app will transcribe it using Whisper.")

# Create buttons for recording control
record_button = st.button("Start Recording")
stop_button = st.button("Stop Recording")
transcription_text = st.empty()  # Placeholder for transcription output


# Function to handle starting the recording
def start_recording():
    st.session_state.recording = True  # Set recording state
    with st.spinner("Recording..."):  # Show spinner while recording
        # Record audio for a maximum of 5 seconds
        st.session_state.audio_data = sd.rec(
            int(SAMPLERATE * 5), samplerate=SAMPLERATE, channels=CHANNELS
        )

        # Wait for the recording to finish
        sd.wait()  # Wait for recording to finish


# Function to handle stopping the recording
def stop_recording():
    st.session_state.recording = False  # Stop recording
    sd.stop()  # Stop the recording
    st.write("Recording stopped.")


# Handle recording logic
if record_button and not st.session_state.recording:
    start_recording()  # Start recording

if stop_button and st.session_state.recording:
    stop_recording()  # Stop recording

    # Transcribe audio after recording
    if st.session_state.audio_data is not None:  # Ensure audio data is available
        transcription = transcribe_audio(
            st.session_state.audio_data
        )  # Transcribe audio
        transcription_text.write(f"**Transcription:** {transcription}")
    else:
        st.error("No audio data available for transcription.")
