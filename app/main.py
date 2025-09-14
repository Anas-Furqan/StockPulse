import streamlit as st
import os
import sys
from dotenv import load_dotenv
import plotly.io as pio
import os





# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import UI components
from app.ui.streamlit_pages import (
    render_overview_page,
    render_eda_page,
    render_anomalies_page,
    render_explainability_page,
    render_model_tuning_page,
    render_export_page,
    render_help_page
)

# Page config
st.set_page_config(
    page_title="StockPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- THEME HANDLING -----------------
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    # st.rerun()

def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0e1117;
                color: white;
            }
            /* Sidebar */
            section[data-testid="stSidebar"] {
                background-color: #1a1d23 !important;
                color: white !important;
            }
            /* Inputs (selectbox, text, textarea) */
            div[data-baseweb="select"] > div {
                background-color: #1a1d23 !important;
                color: white !important;
            }
            input, textarea {
                background-color: #1a1d23 !important;
                color: white !important;
            }
            label, span, p, h1, h2, h3, h4, h5, h6 {
                color: white !important;
            }
            /* Tabs */
            button[data-baseweb="tab"] {
                background-color: #1a1d23 !important;
                color: white !important;
            }
            /* File uploader container */
            .stFileUploader div[role="button"] {
                background-color: #1a1d23 !important;
                color: #ffffff !important;
                border: 1px dashed #444 !important;
                border-radius: 10px;
            }
            /* File uploader label */
            .stFileUploader label {
                color: #ffffff !important;
            }
            /* File uploader Browse button */
            .stFileUploader button {
                background-color: #262730 !important;
                color: #ffffff !important;
                border-radius: 6px;
                border: 1px solid #555 !important;
            }
            /* File uploader drop area */
        .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background-color: #1a1d23 !important;
        border: 2px dashed #444 !important;
        border-radius: 10px;
        color: #ffffff !important;
        }
            /* Dataframes / tables */
            .stDataFrame, .stTable {
                color: #ffffff !important;
                background-color: #1e1e1e !important;
            }
            .dataframe td, .dataframe th {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            /* Buttons */
            .stButton>button {
                background-color: #262730 !important;
                color: #ffffff !important;
                border-radius: 8px;
                border: 1px solid #444 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: white;
                color: black;
            }
            section[data-testid="stSidebar"] {
                background-color: #f9f9f9 !important;
                color: black !important;
            }
            div[data-baseweb="select"] > div {
                background-color: white !important;
                color: black !important;
            }
            input, textarea {
                background-color: white !important;
                color: black !important;
            }
            label, span, p, h1, h2, h3, h4, h5, h6 {
                color: black !important;
            }
            button[data-baseweb="tab"] {
                background-color: #f0f0f0 !important;
                color: black !important;
            }
            .stButton>button {
                background-color: #e0e0e0 !important;
                color: #000000 !important;
                border-radius: 8px;
                border: 1px solid #aaa !important;
            }
            .stFileUploader label {
                color: #000000 !important;
            }
            .stFileUploader div[role="button"] {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px dashed #aaa !important;
                border-radius: 10px;
            }
            .stFileUploader button {
                background-color: #f0f0f0 !important;
                color: #000000 !important;
                border-radius: 6px;
                border: 1px solid #bbb !important;
            }
            .stDataFrame, .stTable {
                color: #000000 !important;
                background-color: #ffffff !important;
            }
            .dataframe td, .dataframe th {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def set_plotly_theme():
    if st.session_state.theme == "dark":
        pio.templates.default = "plotly_dark"
    else:
        pio.templates.default = "plotly_white"

# Apply themes
apply_theme()
set_plotly_theme()

# ----------------- HEADER -----------------
header_col1, header_col2 = st.columns([6, 1])
with header_col1:
    st.title("📊 StockPulse: Smart Anomaly Detection")
    st.caption("AI-powered anomaly detection & explainability for stock markets")
with header_col2:
    st.button("🌓 Toggle Theme", on_click=toggle_theme)

# ----------------- SIDEBAR -----------------
# st.sidebar.title("Navigation")

if "active_page" not in st.session_state:
    st.session_state.active_page = "main"

# ✅ Stock selection (only show on main dashboard)
if st.session_state.active_page == "main":
    st.sidebar.subheader("Quick Stock Selection")
    if 'default_stock' not in st.session_state:
        st.session_state.default_stock = "AAPL"

    default_stock = st.sidebar.selectbox(
        "Select stock symbol:",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM"],
        index=0,
        key="stock_selector"
    )
    if default_stock != st.session_state.default_stock:
        st.session_state.default_stock = default_stock
        if 'data' in st.session_state:
            st.session_state.pop('data')

# ✅ Data source selection (always visible)
if "data_source" not in st.session_state:
    st.session_state.data_source = "Yahoo Finance"

# ✅ Data source selection (always visible)
options = ["Yahoo Finance", "Alpha Vantage", "Upload File"]

if "data_source" not in st.session_state:
    st.session_state.data_source = options[0]  # default "Yahoo Finance"

# Handle mismatches safely
try:
    default_index = options.index(st.session_state.data_source)
except ValueError:
    default_index = 0   # fallback to "Yahoo Finance"
    st.session_state.data_source = options[0]

st.sidebar.subheader("Data Source Options")
selected_source = st.sidebar.radio(
    "Select data source:",
    options,
    index=default_index,
)


# 👉 Detect change in data source
if selected_source != st.session_state.data_source:
    st.session_state.data_source = selected_source
    st.session_state.active_page = "main"   # Force redirect to dashboard
    if 'data' in st.session_state:
        st.session_state.pop('data')
    st.rerun()

# ✅ Sidebar navigation buttons
st.sidebar.subheader("Other Options")
if st.sidebar.button("⚙️ Model Tuning"):
    st.session_state.active_page = "model_tuning"
    st.rerun()
if st.sidebar.button("📤 Export Results"):
    st.session_state.active_page = "export"
    st.rerun()
if st.sidebar.button("❓ Help / About"):
    st.session_state.active_page = "help"
    st.rerun()

# ----------------- MAIN CONTENT -----------------
if st.session_state.active_page == "main":
    tabs = st.tabs(["📊 Overview", "📈 EDA", "🚨 Anomalies", "🔍 Explainability"])

    with tabs[0]:
        render_overview_page(st.session_state.data_source)
    with tabs[1]:
        render_eda_page()
    with tabs[2]:
        render_anomalies_page()
    with tabs[3]:
        render_explainability_page()


elif st.session_state.active_page == "model_tuning":
    render_model_tuning_page()

elif st.session_state.active_page == "export":
    render_export_page()

elif st.session_state.active_page == "help":
    render_help_page()

# Footer
st.markdown("---")
st.markdown("~Made By CodeBits")
