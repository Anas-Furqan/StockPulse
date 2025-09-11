import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import UI components
from app.ui.streamlit_pages import (
    render_overview_page,
    render_eda_page,
    render_anomalies_page,
    render_model_tuning_page,
    render_export_page,
    render_help_page
)

# Set page config
st.set_page_config(
    page_title="StockPulse - Anomaly Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("StockPulse — Anomaly Tracker")
st.markdown("*Developed by Data Science Dominion*")

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "EDA", "Anomalies", "Model Tuning", "Export", "Help"]
)

# Render the selected page
if page == "Overview":
    render_overview_page()
elif page == "EDA":
    render_eda_page()
elif page == "Anomalies":
    render_anomalies_page()
elif page == "Model Tuning":
    render_model_tuning_page()
elif page == "Export":
    render_export_page()
elif page == "Help":
    render_help_page()