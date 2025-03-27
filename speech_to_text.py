import streamlit as st
import speech_recognition as sr
import pyttsx3
import time

# Initialize recognizer and TTS
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# TTS function
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Speech recognition function
def listen_and_transcribe():
    with sr.Microphone() as source:
        st.info("Listening... Please speak")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
        st.info("Processing...")

        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "Could not reach Speech Recognition service."

# Streamlit UI
st.set_page_config(page_title="Speech to Text App", page_icon="🎙️")
st.title("Speech to Text with Voice Feedback")

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

st.write("Click the button to start recording your speech:")

if st.button("Start Listening"):
    transcript = listen_and_transcribe()
    st.session_state.transcript = transcript
    st.success("You said: " + transcript)
    speak_text(transcript)

# Display result
if st.session_state.transcript:
    st.subheader("Transcript:")
    st.write(st.session_state.transcript)
