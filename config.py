import os

# Default local configuration
GEMINI_API_KEY = "AQ.Ab8RN6ID_dpNB31VWYMQeTBPk2m6QKiTME3mGQ1EThFlQuggaQ"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DPI_SCAN = 200
REQUEST_INTERVAL_SECONDS = 2
TEACHER_PIN = "1234"

# If hosted on Streamlit Cloud, prefer st.secrets if configured
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        if "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        if "DEFAULT_MODEL" in st.secrets:
            DEFAULT_MODEL = st.secrets["DEFAULT_MODEL"]
        if "TEACHER_PIN" in st.secrets:
            TEACHER_PIN = str(st.secrets["TEACHER_PIN"])
except Exception:
    pass
