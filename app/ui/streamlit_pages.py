import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
import os
import sys
import base64
import io
import tempfile
import zipfile
import numpy as np
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import data modules
from app.src.data.fetcher import DataFetcher
from app.src.data.loader import DataLoader
from app.src.data.schema import validate_dataframe, infer_column_mapping, apply_column_mapping
import pandas as pd

def auto_fetch_default_stock():
    """
    Automatically fetch data for the default stock symbol.
    """
    try:
        symbol = st.session_state.get('default_stock', 'AAPL')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data_dict = data_fetcher.fetch_from_yahoo(
            tickers=[symbol],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
        if data_dict:
            frames = []
            for sym, df in data_dict.items():
                df_norm = _normalize_df_columns(df, symbol_hint=sym)
                frames.append(df_norm)
            st.session_state.data = pd.concat(frames, ignore_index=True, sort=False)
    except Exception as e:
        st.error(f"Error auto-fetching data: {e}")
        st.session_state.data = None
        
def style_dataframe(df, theme="light"):
    if theme == "dark":
        return df.style.set_properties(
            **{
                "background-color": "#0e1117",
                "color": "white",
                "border-color": "#444"
            }
        ).set_table_styles(
            [
                {"selector": "thead", "props": [("background-color", "#1a1d23"), ("color", "white")]},
                {"selector": "thead th", "props": [("background-color", "#1a1d23"), ("color", "white"), ("border", "1px solid #444")]},
                {"selector": "tbody td", "props": [("background-color", "#0e1117"), ("color", "white"), ("border", "1px solid #444")]}
            ]
        )
    else:
        return df.style.set_properties(
            **{
                "background-color": "white",
                "color": "black",
                "border-color": "#ccc"
            }
        ).set_table_styles(
            [
                {"selector": "thead", "props": [("background-color", "#f0f0f0"), ("color", "black")]},
                {"selector": "thead th", "props": [("background-color", "#f0f0f0"), ("color", "black"), ("border", "1px solid #ccc")]},
                {"selector": "tbody td", "props": [("background-color", "white"), ("color", "black"), ("border", "1px solid #ccc")]}
            ]
        )
        
def _normalize_df_columns(df: pd.DataFrame, symbol_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize dataframe columns:
    - Reset index to get 'Date' if date is index
    - If columns are MultiIndex -> either drop second level (if uniform) or flatten with underscore
    - Rename common columns to standardized names: Date, Open, High, Low, Close, Adj_Close, Volume, Symbol
    - If symbol_hint provided and no Symbol column exists -> add it.
    """
    df = df.copy()

    # 1) Ensure Date column exists (reset index if index is datetime)
    if 'Date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index()
            # try to rename first column to 'Date' if appropriate
            first_col = df.columns[0]
            if str(first_col).lower() in ['index', 'date', 'datetime', 'timestamp']:
                df = df.rename(columns={first_col: 'Date'})

    # 2) Flatten MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        try:
            second_levels = [c[1] for c in df.columns if isinstance(c, tuple)]
            # If second level is uniform (same value) we drop it, else we flatten
            if len(set(second_levels)) == 1:
                df.columns = df.columns.droplevel(1)
            else:
                df.columns = ["_".join([str(p) for p in col if p not in (None, '')]).strip('_') for col in df.columns]
        except Exception:
            df.columns = ["_".join([str(p) for p in col if p not in (None, '')]).strip('_') for col in df.columns]

    # 3) Standardize column names by keyword (not exact match)
    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if 'adj' in lc and 'close' in lc:
            rename_map[c] = 'Adj_Close'
        elif 'close' in lc:
            rename_map[c] = 'Close'
        elif 'open' in lc:
            rename_map[c] = 'Open'
        elif 'high' in lc:
            rename_map[c] = 'High'
        elif 'low' in lc:
            rename_map[c] = 'Low'
        elif 'volume' in lc:
            rename_map[c] = 'Volume'
        elif 'date' in lc:
            rename_map[c] = 'Date'
        elif 'symbol' in lc:
            rename_map[c] = 'Symbol'
    if rename_map:
        df = df.rename(columns=rename_map)

    # 4) Ensure Adj_Close present if Close exists
    if 'Close' in df.columns and 'Adj_Close' not in df.columns:
        df['Adj_Close'] = df['Close']

    # 5) Add Symbol column if hint provided and column missing
    if symbol_hint and 'Symbol' not in df.columns:
        df['Symbol'] = symbol_hint

    return df

def _find_col(df: pd.DataFrame, keyword: str):
    """Return first column name that contains keyword (case-insensitive), else None."""
    keyword = keyword.lower()
    for c in df.columns:
        if keyword in str(c).lower():
            return c
    return None
# --- end helper functions ---
#end of chat code 
# Initialize data fetcher and loader
data_fetcher = DataFetcher()
data_loader = DataLoader()

def render_overview_page(data_source):
    """
    Render the overview page with data source selection and data preview.
    """
    # Project introduction
    st.subheader("Project Introduction")
    st.write("StockPulse is an AI-powered tool for detecting anomalies in stock market data and providing explainable insights.")
   
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = None
        # Auto-fetch default stock data on first load
        if data_source == "Yahoo Finance":
            auto_fetch_default_stock()
    
    # Handle data source selection
    if data_source == "Yahoo Finance":
        render_yahoo_finance_input()
    elif data_source == "Alpha Vantage":
        render_alpha_vantage_input()
    elif data_source == "Upload File":
        render_file_upload()
    
    # Display data preview and metrics if available
    if st.session_state.data is not None:
        df = st.session_state.data

        # Data summary metrics
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)

        # Ensure date column is datetime
        date_col = _find_col(df, 'date')
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])  # remove NaT rows

        # Calculate metrics
        with col1:
            st.metric("Total Records", f"{len(df):,}")

        with col2:
            if date_col and not df.empty:
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                else:
                    date_range = "Date range not available"
                st.metric("Date Range", date_range)
            else:
                st.metric("Date Range", "No valid dates available")

        close_col = _find_col(df, 'close')
        if close_col:
            with col3:
                st.metric("Min Price", f"${df[close_col].min():.2f}")
            with col4:
                st.metric("Max Price", f"${df[close_col].max():.2f}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(style_dataframe(df.head(10), st.session_state.theme), use_container_width=True)
        # render_dataframe(df.head(100), st.session_state.theme)

def render_yahoo_finance_input():
    """
    Render input fields for Yahoo Finance data source.
    """
    st.subheader("Yahoo Finance Data")
    
    # Input for ticker symbols
    ticker_input = st.text_input("Enter ticker symbols (comma-separated):", "AAPL")
    tickers = [ticker.strip() for ticker in ticker_input.split(",")]
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date:", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End date:", datetime.now())
    
    # Interval selection
    interval = st.selectbox(
        "Select interval:",
        ["1d", "1h", "5m"],
        index=0
    )
    
    # Fetch data button
    if st.button("Fetch Data"):
        with st.spinner("Fetching data from Yahoo Finance..."):
            try:
                data_dict = data_fetcher.fetch_from_yahoo(
                    tickers=tickers,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    interval=interval
                )
                
                # Combine data from all tickers
                if data_dict:
                    frames = []
                    for sym, df in data_dict.items():
                        # normalize each dataframe and add Symbol
                        df_norm = _normalize_df_columns(df, symbol_hint=sym)
                        frames.append(df_norm)
                    # concat all rows into single table
                    st.session_state.data = pd.concat(frames, ignore_index=True, sort=False)
                    st.success(f"Successfully fetched data for {', '.join(data_dict.keys())}")
                else:
                    st.error("No data fetched. Please check your inputs.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

def render_alpha_vantage_input():
    """
    Render input fields for Alpha Vantage data source.
    """
    st.subheader("Alpha Vantage Data")
    
    # Check if API key is available
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        api_key = st.text_input("Enter Alpha Vantage API key:")
        if not api_key:
            st.warning("Please enter an Alpha Vantage API key to proceed.")
            return
    
    # Input for ticker symbols
    ticker_input = st.text_input("Enter ticker symbols (comma-separated):", "AAPL")
    tickers = [ticker.strip() for ticker in ticker_input.split(",")]
    
    # Output size selection
    output_size = st.selectbox(
        "Select output size:",
        ["compact", "full"],
        index=0,
        help="'compact' returns the latest 100 data points, 'full' returns up to 20 years of historical data"
    )
    
    # Interval selection
    interval = st.selectbox(
        "Select interval:",
        ["daily", "weekly", "monthly", "intraday"],
        index=0
    )
    
    # Fetch data button
    if st.button("Fetch Data"):
        with st.spinner("Fetching data from Alpha Vantage..."):
            try:
                data_dict = data_fetcher.fetch_from_alpha_vantage(
                    tickers=tickers,
                    output_size=output_size,
                    interval=interval
                )
                
                # Combine data from all tickers
                if data_dict:
                    st.session_state.data = pd.concat(list(data_dict.values()), ignore_index=True)
                    st.success(f"Successfully fetched data for {', '.join(data_dict.keys())}")
                else:
                    st.error("No data fetched. Please check your inputs.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                

def render_file_upload():
    """
    Render file upload for CSV, JSON, and XML data sources.
    """
    st.subheader("Upload Data File")

    uploaded_file = st.file_uploader(
        "Upload a CSV, JSON, or XML file",
        type=["csv", "json", "xml"],
        help="Upload your stock data file in CSV, JSON, or XML format containing stock price data with at least Date and Close price columns"
    )

    if uploaded_file is None:
        st.info("Please upload a file using the uploader above.")
        return
    

    try:
        # Save uploaded file to a temporary location
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            file_path = temp_file.name

        # Load data based on extension
        if file_extension == "csv":
            df = pd.read_csv(file_path)
        elif file_extension == "json":
            try:
                df = pd.read_json(file_path)
            except Exception:
                df = parse_json_file(file_path)
        elif file_extension == "xml":
            df = parse_xml_file(file_path)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return

        if df is None or df.empty:
            st.error("Uploaded file is empty or could not be loaded. Please check your file.")
            return

        # Show file info
        st.success(f"File loaded successfully! {len(df)} rows, {len(df.columns)} columns")
        st.write("Available columns:", ", ".join(df.columns))

        # ---- Column Mapping Section ----
        column_mapping = infer_column_mapping(df)
        st.subheader("Column Mapping")
        st.write("Map your columns to required format:")
        st.write("**Required columns:** Date, Close")

        col1, col2 = st.columns(2)
        new_mapping = {}

        # Required Columns
        with col1:
            st.markdown("**Required Columns**")

            date_options = ["None"] + [col for col in df.columns if any(k in col.lower() for k in ["date", "time"])]
            date_default = column_mapping.get("Date", date_options[1] if len(date_options) > 1 else "None")
            new_mapping["Date"] = st.selectbox("Date Column", date_options, index=date_options.index(date_default) if date_default in date_options else 0)

            close_options = ["None"] + [col for col in df.columns if any(k in col.lower() for k in ["close", "price", "last"])]
            close_default = column_mapping.get("Close", close_options[1] if len(close_options) > 1 else "None")
            new_mapping["Close"] = st.selectbox("Close Price Column", close_options, index=close_options.index(close_default) if close_default in close_options else 0)

        # Optional Columns
        with col2:
            st.markdown("**Optional Columns**")

            for col_name, keywords in {
                "Open": ["open"],
                "High": ["high"],
                "Low": ["low"],
                "Volume": ["volume", "vol"],
            }.items():
                options = ["None"] + [col for col in df.columns if any(k in col.lower() for k in keywords)]
                default_val = column_mapping.get(col_name, options[1] if len(options) > 1 else "None")
                new_mapping[col_name] = st.selectbox(
                    f"{col_name} Column", options, index=options.index(default_val) if default_val in options else 0
                )

        # Remove "None" selections
        new_mapping = {k: v for k, v in new_mapping.items() if v != "None"}

        if new_mapping:
            st.info("Current mapping: " + ", ".join([f"{k} → {v}" for k, v in new_mapping.items()]))
        else:
            st.warning("No columns mapped yet. Please select at least Date and Close columns.")

        st.subheader("Data Processing Options")
        auto_clean = st.checkbox("Auto-clean & preprocess", value=True)
        auto_predict = st.checkbox("Auto-detect anomalies", value=True)

        # ---- Process Data ----
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                try:
                    # Ensure required columns are mapped
                    required = ["Date", "Close"]
                    missing = [col for col in required if col not in new_mapping]
                    if missing:
                        st.error(f"Missing required columns: {', '.join(missing)}")
                        return

                    # Apply mapping
                    mapped_df = df.rename(columns={v: k for k, v in new_mapping.items()})

                    # Convert Date column
                    mapped_df["Date"] = pd.to_datetime(mapped_df["Date"], errors="coerce")
                    mapped_df = mapped_df.dropna(subset=["Date"])
                    if mapped_df.empty:
                        st.error("No valid dates found. Please check your data.")
                        return

                    # Convert numeric columns (handle Close/Last gracefully)
                    for col in ["Close", "Open", "High", "Low", "Volume"]:
                        if col in mapped_df.columns:
                            mapped_df[col] = (
                                mapped_df[col]
                                .astype(str)
                                .str.replace(",", "", regex=False)
                                .str.replace("$", "", regex=False)
                                .str.strip()
                            )
                            mapped_df[col] = pd.to_numeric(mapped_df[col], errors="coerce")

                    # Check Close column again
                    nan_count = mapped_df["Close"].isna().sum()
                    if nan_count > 0:
                        st.warning(f"{nan_count} rows had invalid Close values and were removed.")
                        mapped_df = mapped_df.dropna(subset=["Close"])

                    if mapped_df.empty:
                        st.error("No valid Close prices found after cleaning. Please check your data.")
                        return

                    # Auto-clean
                    if auto_clean:
                        for col in mapped_df.select_dtypes(include=[np.number]).columns:
                            mapped_df[col] = mapped_df[col].fillna(method="ffill").fillna(method="bfill")

                        if "Adj_Close" not in mapped_df.columns and "Close" in mapped_df.columns:
                            mapped_df["Adj_Close"] = mapped_df["Close"]

                        if "Symbol" not in mapped_df.columns:
                            mapped_df["Symbol"] = uploaded_file.name.split(".")[0]

                    mapped_df = mapped_df.sort_values("Date").reset_index(drop=True)

                    # Data quality check
                    st.write("### Data Quality Check")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Total Rows", len(mapped_df))
                        st.metric("Date Range", f"{mapped_df['Date'].min().date()} → {mapped_df['Date'].max().date()}")
                    with c2:
                        st.metric("Missing Close Values", int(mapped_df["Close"].isna().sum()))
                        if "Volume" in mapped_df.columns:
                            st.metric("Missing Volume Values", int(mapped_df["Volume"].isna().sum()))

                    st.session_state.data = mapped_df
                    st.session_state.data_source = "file_upload"
                    st.success(f"Data processed successfully! {len(mapped_df)} rows ready for analysis.")

                    # ---- Anomaly Detection ----
                    if auto_predict:
                        try:
                            from app.src.models import create_model

                            numeric_cols = mapped_df.select_dtypes(include=[np.number]).columns.tolist()
                            numeric_cols = [c for c in numeric_cols if c not in ["anomaly_score", "is_anomaly"]]

                            model = create_model("isolation_forest", {
                                "contamination": 0.1,
                                "n_estimators": 100,
                                "random_state": 42
                            })

                            # feature_data = mapped_df[numeric_cols].fillna(mapped_df[numeric_cols].mean())
                            # model.fit(feature_data)

                            # mapped_df["anomaly_score"] = model.decision_function(feature_data)
                            # mapped_df["is_anomaly"] = (model.predict(feature_data) == -1).astype(int)
                            # Fit the wrapper model
                            model.fit(mapped_df, numeric_cols)

                            # Predict anomalies (this adds anomaly_score & is_anomaly)
                            mapped_df = model.predict(mapped_df)
                            st.session_state.anomaly_results = mapped_df
                            st.session_state.anomaly_model = model
                            st.session_state.anomaly_model_type = "isolation_forest"
                            st.session_state.anomaly_features = numeric_cols

                            st.success(f"Detected {mapped_df['is_anomaly'].sum()} potential anomalies!")

                        except Exception as e:
                            st.warning(f"Anomaly detection failed: {e}")

                    # ---- Data Preview ----
                    tabs = st.tabs(["Data Preview", "Statistics", "Column Info"])
                    with tabs[0]:
                        st.dataframe(mapped_df.head(20))
                    with tabs[1]:
                        st.dataframe(mapped_df.describe())
                    with tabs[2]:
                        col_info = pd.DataFrame({
                            "Column": mapped_df.columns,
                            "Type": mapped_df.dtypes,
                            "Non-Null Count": mapped_df.count(),
                            "Null Count": mapped_df.isna().sum(),
                            "Null %": (mapped_df.isna().sum() / len(mapped_df) * 100).round(2)
                        })
                        st.dataframe(col_info)

                except Exception as e:
                    st.error(f"Error during data processing: {e}")

        # Clean up temp file
        try:
            os.remove(file_path)
        except Exception:
            pass

    except Exception as e:
        import traceback
        st.error(f"Error loading file: {e}")
        st.error(traceback.format_exc())


def parse_json_file(file_path):
    """
    Parse a JSON file into a pandas DataFrame.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of records
            return pd.json_normalize(data)
        elif isinstance(data, dict):
            # Check if it's a nested structure
            if any(isinstance(v, (dict, list)) for v in data.values()):
                # Try to find the main data array
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        return pd.json_normalize(value)
                # If no suitable array found, flatten the structure
                return pd.json_normalize(data)
            else:
                # Single record
                return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return pd.DataFrame()

def parse_xml_file(file_path):
    """
    Parse an XML file into a pandas DataFrame.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find all child elements that might be data records
        data_elements = []
        
        # Try to identify record elements (look for repeating elements)
        element_counts = {}
        for child in root.iter():
            tag = child.tag
            element_counts[tag] = element_counts.get(tag, 0) + 1
        
        # Find the most common repeating element that might be a record
        record_candidates = [tag for tag, count in element_counts.items() if count > 1]
        
        if record_candidates:
            # Use the first candidate as the record element
            record_tag = record_candidates[0]
            records = []
            
            for record in root.iter(record_tag):
                record_data = {}
                for child in record:
                    record_data[child.tag] = child.text
                records.append(record_data)
            
            return pd.DataFrame(records)
        else:
            # Fallback: treat the root's direct children as a single record
            record_data = {}
            for child in root:
                record_data[child.tag] = child.text
            return pd.DataFrame([record_data])
    except Exception as e:
        st.error(f"Error parsing XML: {e}")
        return pd.DataFrame()

def render_eda_page():
    """
    Render the EDA page with data analysis and visualizations.
    """
    st.subheader("📈 Exploratory Data Analysis")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Overview page.")
        return

    # Use a copy of the session data and normalize top-level columns
    df_all = _normalize_df_columns(st.session_state.data.copy())
    
    # Apply theme based on session state
    theme = st.session_state.get('theme', 'dark')
    plotly_template = "plotly_dark" if theme == "dark" else "plotly_white"
    
    # Filter by symbol if multiple symbols exist
    if 'Symbol' in df_all.columns and len(df_all['Symbol'].unique()) > 1:
        symbols = df_all['Symbol'].unique()
        selected_symbol = st.selectbox("Select symbol:", symbols)
        df = df_all[df_all['Symbol'] == selected_symbol].copy()
    else:
        df = df_all.copy()
    
    # Ensure df columns normalized
    df = _normalize_df_columns(df)

    # Detect date/close/volume columns by keyword (robust to variations)
    date_col = _find_col(df, 'date')
    open_col = _find_col(df, 'open')
    high_col = _find_col(df, 'high')
    low_col = _find_col(df, 'low')
    close_col = _find_col(df, 'close')
    volume_col = _find_col(df, 'volume')

    # Ensure date column is datetime
    if date_col and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    # Sort by date
    if date_col:
        df = df.sort_values(by=date_col)

    # x values (date or index)
    if date_col:
        x_values = df[date_col]
    else:
        x_values = df.index
    
    # Calculate returns if close price is available
    if close_col:
        # Ensure close_col is numeric
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        
        # Calculate returns only if we have valid numeric data
        if not df[close_col].isna().all():
            df['Returns'] = df[close_col].pct_change() * 100
            
            # Calculate moving averages
            df['MA7'] = df[close_col].rolling(window=7).mean()
            df['MA30'] = df[close_col].rolling(window=30).mean()
        else:
            st.warning(f"Could not convert '{close_col}' column to numeric values. Some visualizations may not be available.")
    
    # Create tabs for different visualizations
    eda_tabs = st.tabs(["Price Chart", "Volume", "Returns", "Statistics"])
    
    # Tab 1: Price Chart with Moving Averages
    with eda_tabs[0]:
        if close_col is not None and not df[close_col].isna().all():
            st.subheader("Price Time Series with Moving Averages")
            
            # Create figure
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=x_values, 
                y=df[close_col], 
                mode='lines', 
                name='Close Price',
                line=dict(color='#2196F3', width=2)
            ))
            
            # Add moving averages if they were calculated
            if 'MA7' in df.columns:
                fig.add_trace(go.Scatter(
                    x=x_values, 
                    y=df['MA7'], 
                    mode='lines', 
                    name='7-day MA',
                    line=dict(color='#FF9800', width=1.5, dash='dash')
                ))
            
            if 'MA30' in df.columns:
                fig.add_trace(go.Scatter(
                    x=x_values, 
                    y=df['MA30'], 
                    mode='lines', 
                    name='30-day MA',
                    line=dict(color='#4CAF50', width=1.5, dash='dash')
                ))
            
            # Update layout
            fig.update_layout(
                template=plotly_template,
                title="Price Time Series with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="white" if st.session_state.theme == "dark" else "black")
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
            
            # Add custom hover template
            fig.update_traces(
                hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<br>"
            )

            if st.session_state.theme == "dark":
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white")
             )
            else:
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black")
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Returns data is not available. This could be due to missing or non-numeric 'Close' values.")
    
    # Tab 2: Volume Chart
    with eda_tabs[1]:
        if volume_col is not None and not df[volume_col].isna().all():
            st.subheader("Trading Volume")
            
            # Create figure
            fig = go.Figure()
            
            # Add volume bars
            fig.add_trace(go.Bar(
                x=x_values, 
                y=df[volume_col], 
                name='Volume',
                marker=dict(color='rgba(158, 158, 158, 0.6)')
            ))
            
            # Update layout
            fig.update_layout(
                template=plotly_template,
                title="Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                hovermode="x unified",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
            
            # Add custom hover template
            fig.update_traces(
                hovertemplate="<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<br>"
            )
            
            if st.session_state.theme == "dark":
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white")
             )
            else:
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black")
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No 'Volume' column found in the data.")
    
    # Tab 3: Returns Plot
    with eda_tabs[2]:
        if 'Returns' in df.columns and not df['Returns'].isna().all():
            st.subheader("Daily Returns (%)")
            
            # Create figure
            fig = go.Figure()
            
            # Add returns line
            fig.add_trace(go.Bar(
                x=x_values, 
                y=df['Returns'], 
                name='Daily Returns',
                marker=dict(
                    color=np.where(df['Returns'] >= 0, '#4CAF50', '#F44336')
                )
            ))
            
            # Update layout
            fig.update_layout(
                template=plotly_template,
                title="Daily Returns (%)",
                xaxis_title="Date",
                yaxis_title="Returns (%)",
                hovermode="x unified",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
            
            # Add custom hover template
            fig.update_traces(
                hovertemplate="<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<br>"
            )
            
            if st.session_state.theme == "dark":
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white")
             )
            else:
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black")
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Price chart is not available. This could be due to missing or non-numeric 'Close' values.")
    
    # Tab 4: Statistics
    with eda_tabs[3]:
        st.subheader("Statistical Summary")
        
        # Create columns for statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Display basic statistics
            st.write("Descriptive Statistics")
            # render_dataframe(df.describe(), st.session_state.theme)
            st.dataframe(style_dataframe(df.describe().T, st.session_state.theme), use_container_width=True)
            
        with col2:
            # Display missing values
            st.write("Missing Values")
            missing_data = pd.DataFrame({
                'Column': df.isnull().sum().index,
                'Missing Values': df.isnull().sum().values,
                'Percentage': (df.isnull().sum() / len(df) * 100).values
            })
            # render_dataframe(missing_data, st.session_state.theme)
            st.dataframe(style_dataframe(missing_data, st.session_state.theme), use_container_width=True)
            
            # Display data types
            st.write("Data Types")
            st.dataframe(style_dataframe(pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values
            }), st.session_state.theme), use_container_width=True)

def render_anomalies_page():
    """
    Render the anomalies page with anomaly detection results.
    """
    st.subheader("Anomaly Detection Results")

    result_df = pd.DataFrame()  

    if "anomaly_results" in st.session_state:
        result_df = st.session_state.anomaly_results

        anomalies = result_df[result_df['is_anomaly'] == 1].sort_values(
            by='anomaly_score', ascending=False
        )

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies.head(20))
    else:
        st.warning("No anomaly results available. Please run anomaly detection first.")

    
    # Import models
    from app.src.models import create_model
    
    # Check if data exists in session state and is not empty
    if 'data' not in st.session_state or st.session_state.data is None or len(st.session_state.data) == 0:
        st.error("No data available for anomaly detection. Please upload and process a file, or fetch data from Yahoo Finance or Alpha Vantage.")
        return
    # Get the data
    df = st.session_state.data.copy()
    
    # Apply theme based on session state
    theme = st.session_state.get('theme', 'dark')
    plotly_template = "plotly_dark" if theme == "dark" else "plotly_white"
    
    # Create layout with columns
    col_config, col_results = st.columns([1, 2])
    
    with col_config:
        st.markdown("### Model Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select model:",
            ["isolation_forest", "lof", "autoencoder"],
            index=0,
            format_func=lambda x: {
                "isolation_forest": "Isolation Forest",
                "lof": "Local Outlier Factor",
                "autoencoder": "Autoencoder Neural Network"
            }.get(x, x)
        )
        
        # Common parameters
        contamination = st.slider(
            "Contamination (expected % of anomalies):",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.2f"
        )
        
        # Model-specific parameters
        if model_type == "isolation_forest":
            n_estimators = st.slider(
                "Number of estimators:",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )
            params = {
                "contamination": contamination,
                "n_estimators": n_estimators,
                "random_state": 42
            }
        
        elif model_type == "lof":
            n_neighbors = st.slider(
                "Number of neighbors:",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
            params = {
                "contamination": contamination,
                "n_neighbors": n_neighbors
            }
        
        elif model_type == "autoencoder":
            encoding_dim = st.slider(
                "Encoding dimension:",
                min_value=2,
                max_value=20,
                value=10,
                step=1
            )
            epochs = st.slider(
                "Training epochs:",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )
            batch_size = st.slider(
                "Batch size:",
                min_value=8,
                max_value=128,
                value=32,
                step=8
            )
            params = {
                "contamination": contamination,
                "encoding_dim": encoding_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "random_state": 42
            }
        
        # Feature selection
        st.markdown("### Feature Selection")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove date and anomaly columns if they exist
        feature_columns = [col for col in numeric_columns if col not in ['anomaly_score', 'is_anomaly']]
        
        selected_features = st.multiselect(
            "Select features for anomaly detection:",
            options=feature_columns,
            default=feature_columns
        )
        
        if not selected_features:
            st.warning("Please select at least one feature.")
            return
        
        # Run anomaly detection
        run_button = st.button("Detect Anomalies", type="primary")
    
    with col_results:
        # Check if we should run detection or display previous results
        run_detection = run_button or ('anomaly_results' in st.session_state)
        
        if run_detection:
            # If button was clicked, run new detection
            if run_button:
                with st.spinner(f"Running {model_type.replace('_', ' ').title()} model..."):
                    try:
                        # Check if we have data and selected features
                        if len(df) == 0:
                            raise ValueError("No data available for anomaly detection. Please ensure data is loaded properly.")
                        
                        if len(selected_features) == 0:
                            raise ValueError("No features selected for anomaly detection. Please select at least one feature.")
                        
                        # Create and fit the model
                        model = create_model(model_type, params)
                        model.fit(df, selected_features)
                        
                        # Predict anomalies
                        result_df = model.predict(df)
                        
                        # Store results in session state
                        st.session_state.anomaly_results = result_df
                        st.session_state.anomaly_model = model
                        st.session_state.anomaly_model_type = model_type
                        st.session_state.anomaly_features = selected_features
                        
                        st.success("Anomaly detection completed successfully!")
                    except Exception as e:
                        st.error(f"Error during anomaly detection: {e}")
                        st.exception(e)
                        return
            else:
                # Use previous results
                result_df = st.session_state.anomaly_results
                
            # Display results with tabs
            anomaly_count = result_df['is_anomaly'].sum()
            total_count = len(result_df)
            anomaly_percentage = (anomaly_count / total_count) * 100
            
            # Create metrics row
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Total Data Points", total_count)
            with metric_cols[1]:
                st.metric("Anomalies Detected", anomaly_count, delta=f"{anomaly_percentage:.1f}%")
            with metric_cols[2]:
                st.metric("Normal Points", total_count - anomaly_count)
            
            # Create tabs for different visualizations
            anomaly_tabs = st.tabs(["Price Chart", "Volume Chart", "Anomaly Table", "Score Distribution"])
            
            # Normalize column names for consistency
            result_df = _normalize_df_columns(result_df)
            
            # Find date and price columns
            date_col = _find_col(result_df, 'date')
            close_col = _find_col(result_df, 'close')
            volume_col = _find_col(result_df, 'volume')
            
            # Ensure date column is datetime
            if date_col and not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Tab 1: Price Chart with Anomalies
            with anomaly_tabs[0]:
                if date_col and close_col:
                    # Create figure
                    fig = go.Figure()
                    
                    # Add price line for all points
                    fig.add_trace(go.Scatter(
                        x=result_df[date_col],
                        y=result_df[close_col],
                        mode='lines',
                        name='Price',
                        line=dict(color='#2196F3', width=2)
                    ))
                    
                    # Add normal points
                    normal_points = result_df[result_df['is_anomaly'] == 0]
                    fig.add_trace(go.Scatter(
                        x=normal_points[date_col],
                        y=normal_points[close_col],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='#4CAF50', size=8)
                    ))
                    
                    # Add anomaly points
                    anomaly_points = result_df[result_df['is_anomaly'] == 1]
                    fig.add_trace(go.Scatter(
                        x=anomaly_points[date_col],
                        y=anomaly_points[close_col],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='#F44336', size=10)
                    ))
                    
                    fig.update_layout(
                        template=plotly_template,
                        title="Price Chart with Anomalies Highlighted",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="closest",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white" if st.session_state.theme == "dark" else "black")),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    # Add custom hover template
                    fig.update_traces(
                        hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<br>"
                    )
                    
                    # Add range slider
                    fig.update_xaxes(rangeslider_visible=True)
                    
                    if st.session_state.theme == "dark":
                        fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white")
                    )
                    else:
                        fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Cannot create price chart: missing date or close price columns.")
            
            # Tab 2: Volume Chart with Anomalies
            with anomaly_tabs[1]:
                if date_col and volume_col:
                    # Create figure
                    fig = go.Figure()
                    
                    # Add volume bars for all points
                    fig.add_trace(go.Bar(
                        x=result_df[date_col],
                        y=result_df[volume_col],
                        name='Volume',
                        marker=dict(color='rgba(158, 158, 158, 0.6)')
                    ))
                    
                    # Add normal points
                    normal_points = result_df[result_df['is_anomaly'] == 0]
                    fig.add_trace(go.Scatter(
                        x=normal_points[date_col],
                        y=normal_points[volume_col],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='#4CAF50', size=8)
                    ))
                    
                    # Add anomaly points
                    anomaly_points = result_df[result_df['is_anomaly'] == 1]
                    fig.add_trace(go.Scatter(
                        x=anomaly_points[date_col],
                        y=anomaly_points[volume_col],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='#F44336', size=10)
                    ))
                    
                    fig.update_layout(
                        template=plotly_template,
                        title="Volume Chart with Anomalies Highlighted",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        hovermode="closest",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white" if st.session_state.theme == "dark" else "black")),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    # Add custom hover template
                    fig.update_traces(
                        hovertemplate="<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<br>"
                    )
                    
                    # Add range slider
                    fig.update_xaxes(rangeslider_visible=True)
                    
                    if st.session_state.theme == "dark":
                        fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white")
                    )
                    else:
                        fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Cannot create volume chart: missing date or volume columns.")
            
            # Tab 3: Anomaly Table
            with anomaly_tabs[2]:
                st.subheader("Anomaly Details")
                
                # Display anomaly data table with sorting
                anomalies = result_df[result_df['is_anomaly'] == 1].sort_values(by='anomaly_score', ascending=False)
                
                # Format the table for better readability
                display_cols = [date_col] if date_col else []
                if close_col:
                    display_cols.append(close_col)
                if volume_col:
                                 display_cols.append(volume_col)
                display_cols.append('anomaly_score')
                
                # Add selected features but avoid duplicates
                for feature in selected_features:
                    if feature not in display_cols:  # Avoid duplicates
                        display_cols.append(feature)
                
                # Create a copy of the dataframe with only the columns we want to display
                display_df = anomalies[display_cols].copy()
                
                # Convert datetime columns to string to avoid Arrow conversion issues
                for col in display_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                        display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Only show relevant columns
                if not display_df.empty:
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("No anomalies detected.")
            
            # Tab 4: Anomaly Score Distribution
            with anomaly_tabs[3]:
                # Plot anomaly score distribution
                fig = px.histogram(
                    result_df,
                    x='anomaly_score',
                    color='is_anomaly',
                    nbins=50,
                    title='Anomaly Score Distribution',
                    labels={'is_anomaly': 'Is Anomaly', 'anomaly_score': 'Anomaly Score'},
                    color_discrete_map={0: '#4CAF50', 1: '#F44336'}
                )
                
                fig.update_layout(
                    template=plotly_template,
                    xaxis_title="Anomaly Score",
                    yaxis_title="Count",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white" if st.session_state.theme == "dark" else "black")),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if st.session_state.theme == "dark":
                    fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white")
                    )
                else:
                    fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black")
                    )
        else:
            # Show placeholder when no detection has been run
            st.info("👈 Configure the anomaly detection model and click 'Detect Anomalies' to start.")
            
            # Show example of what to expect
            st.markdown("### What to expect")
            st.markdown("""
            The anomaly detection will:
            - Analyze your data using the selected model
            - Highlight unusual patterns in red
            - Show normal data points in green
            - Provide detailed anomaly scores
            - Generate explanations for detected anomalies
            """)
            
            # Add sample visualization placeholder
            st.image("https://miro.medium.com/max/1400/1*-NCXnUXNF_lOqJnKN_20Pw.png", use_column_width=True)
        
        # Display anomalies table
        if not result_df.empty and "is_anomaly" in result_df.columns:
            st.subheader("Anomalies")
            anomalies = result_df[result_df['is_anomaly'] == 1].sort_values(
                by='anomaly_score', ascending=False
            )
            st.dataframe(anomalies)

            st.subheader("Anomaly Visualization")
            from app.ui.components import plot_anomalies

            if "Date" in result_df.columns and "Close" in result_df.columns:
                plot_anomalies(
                    result_df,
                    x_col="Date",
                    y_col="Close",
                    anomaly_col="is_anomaly",
                    title="Price Anomalies"
                )

            if "Date" in result_df.columns and "Volume" in result_df.columns:
                plot_anomalies(
                    result_df,
                    x_col="Date",
                    y_col="Volume",
                    anomaly_col="is_anomaly",
                    title="Volume Anomalies"
                )
        else:
            st.info("No anomaly results to display yet.")

def render_explainability_page():
    """
    Render the explainability page with explanations for detected anomalies.
    """
    st.subheader("🔍 Anomaly Explainability")
    
    # Check if anomaly results are available
    if 'anomaly_results' not in st.session_state or st.session_state.anomaly_results is None:
        st.warning("Please run anomaly detection first on the Anomalies page.")
        return
    
    # Apply theme based on session state
    theme = st.session_state.get('theme', 'dark')
    plotly_template = "plotly_dark" if theme == "dark" else "plotly_white"
    
    # Get the anomaly results
    result_df = st.session_state.anomaly_results.copy()
    model_type = st.session_state.get('anomaly_model_type', 'unknown')
    selected_features = st.session_state.get('anomaly_features', [])
    
    # Normalize column names for consistency
    result_df = _normalize_df_columns(result_df)
    
    # Find date and price columns
    date_col = _find_col(result_df, 'date')
    close_col = _find_col(result_df, 'close')
    volume_col = _find_col(result_df, 'volume')
    
    # Ensure date column is datetime
    if date_col and not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
        result_df[date_col] = pd.to_datetime(result_df[date_col])
    
    # Get anomalies
    anomalies = result_df[result_df['is_anomaly'] == 1].sort_values(by='anomaly_score', ascending=False)
    
    if len(anomalies) == 0:
        st.info("No anomalies were detected in the data.")
        return
    
    # Display summary
    st.markdown(f"### Explaining {len(anomalies)} anomalies detected by {model_type.replace('_', ' ').title()}")
    
    # Create explanation cards for anomalies
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a figure for the timeline of anomalies
        if date_col and close_col:
            fig = go.Figure()
            
            # Add price line for all points
            fig.add_trace(go.Scatter(
                x=result_df[date_col],
                y=result_df[close_col],
                mode='lines',
                name='Price',
                line=dict(color='rgba(33, 150, 243, 0.3)', width=1)
            ))
            
            # Add anomaly points
            fig.add_trace(go.Scatter(
                x=anomalies[date_col],
                y=anomalies[close_col],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#F44336', size=10, symbol='circle')
            ))
            
            fig.update_layout(
                template=plotly_template,
                title="Timeline of Detected Anomalies",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white" if st.session_state.theme == "dark" else "black")),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            if st.session_state.theme == "dark":
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white")
                    )
            else:
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black")
                    )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display summary statistics about anomalies
        st.markdown("### Anomaly Patterns")
        
        # Calculate some statistics about when anomalies occur
        if date_col:
            # Check if anomalies are clustered
            anomaly_dates = anomalies[date_col].sort_values()
            if len(anomaly_dates) > 1:
                date_diffs = anomaly_dates.diff().dropna()
                avg_days_between = date_diffs.mean().days if hasattr(date_diffs.mean(), 'days') else 0
                max_cluster = date_diffs[date_diffs.dt.days < 3].count() if hasattr(date_diffs, 'dt') else 0
                
                st.metric("Avg. Days Between Anomalies", f"{avg_days_between:.1f}")
                st.metric("Largest Anomaly Cluster", max_cluster)
        
        # Check if anomalies are related to high/low values
        if close_col:
            price_percentile = np.percentile(result_df[close_col], [25, 50, 75, 90, 95])
            high_price_anomalies = anomalies[anomalies[close_col] > price_percentile[3]].shape[0]
            low_price_anomalies = anomalies[anomalies[close_col] < price_percentile[0]].shape[0]
            
            high_pct = (high_price_anomalies / len(anomalies)) * 100
            low_pct = (low_price_anomalies / len(anomalies)) * 100
            
            st.metric("High Price Anomalies", f"{high_pct:.1f}%")
            st.metric("Low Price Anomalies", f"{low_pct:.1f}%")
    
    # Create cards for each anomaly with explanations
    st.markdown("### Detailed Anomaly Explanations")
    
    # Function to generate explanation for an anomaly
    def explain_anomaly(row, features):
        explanations = []
        confidence = "High" if row['anomaly_score'] > 0.8 else "Medium" if row['anomaly_score'] > 0.6 else "Low"
        
        # Check for price-related anomalies
        if close_col in features:
            # Get percentile of this price
            price = row[close_col]
            price_percentile_val = percentileofscore(result_df[close_col], price)
            
            if price_percentile_val > 95:
                explanations.append(f"Unusually high price (top {100-price_percentile_val:.1f}%)")
            elif price_percentile_val < 5:
                explanations.append(f"Unusually low price (bottom {price_percentile_val:.1f}%)")
        
        # Check for volume-related anomalies
        if volume_col in features:
            volume = row[volume_col]
            volume_percentile = percentileofscore(result_df[volume_col], volume)
            
            if volume_percentile > 95:
                explanations.append(f"Extremely high trading volume (top {100-volume_percentile:.1f}%)")
            elif volume_percentile < 5:
                explanations.append(f"Unusually low trading volume (bottom {volume_percentile:.1f}%)")
        
        # Check for volatility (if we have OHLC data)
        high_col = _find_col(result_df, 'high')
        low_col = _find_col(result_df, 'low')
        
        if high_col and low_col and high_col in features and low_col in features:
            daily_range = row[high_col] - row[low_col]
            avg_range = (result_df[high_col] - result_df[low_col]).mean()
            
            if daily_range > 2 * avg_range:
                explanations.append(f"High volatility (daily range {daily_range/avg_range:.1f}x normal)")
        
        # If no specific explanations, provide a generic one
        if not explanations:
            explanations.append("Unusual pattern detected across multiple features")
        
        return {
            "explanations": explanations,
            "confidence": confidence
        }
    
    # Calculate percentile scores for statistical comparisons
    from scipy.stats import percentileofscore
    
    # Display cards for top anomalies
    top_anomalies = anomalies.head(10)  # Limit to top 10 anomalies
    
    for i, (idx, row) in enumerate(top_anomalies.iterrows()):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            date_str = row[date_col].strftime("%Y-%m-%d") if date_col and hasattr(row[date_col], 'strftime') else "Unknown Date"
            price_str = f"${row[close_col]:.2f}" if close_col else "Unknown Price"
            score = row['anomaly_score']
            
            explanation = explain_anomaly(row, selected_features)
            
            with col1:
                if score > 0.8:
                    st.error(f"**Anomaly #{i+1}**\n{date_str}\n{price_str}\nScore: {score:.2f}")
                elif score > 0.6:
                    st.warning(f"**Anomaly #{i+1}**\n{date_str}\n{price_str}\nScore: {score:.2f}")
                else:
                    st.info(f"**Anomaly #{i+1}**\n{date_str}\n{price_str}\nScore: {score:.2f}")
            
            with col2:
                st.markdown(f"**Confidence: {explanation['confidence']}**")
                for exp in explanation['explanations']:
                    st.markdown(f"• {exp}")
                
                # Add contextual information if available
                if date_col and close_col:
                    # Get data points before and after this anomaly
                    date_idx = result_df[result_df[date_col] == row[date_col]].index[0]
                    
                    try:
                        prev_idx = date_idx - 1
                        next_idx = date_idx + 1
                        
                        if prev_idx >= 0 and prev_idx < len(result_df):
                            prev_price = result_df.iloc[prev_idx][close_col]
                            price_change = ((row[close_col] - prev_price) / prev_price) * 100
                            st.markdown(f"• Price change: **{price_change:.2f}%** from previous day")
                    except Exception:
                        pass
                    except:
                        pass
            
            st.markdown("---")

def render_model_tuning_page():
    """
    Render the model tuning page with hyperparameter tuning options.
    """
    st.header("Model Tuning")
    
    # Check if data is available
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data from the Overview page first.")
        return
    
    # Import models
    from app.src.models import create_model
    
    # Get the data
    df = st.session_state.data
    
    # Model selection
    st.subheader("Select Model to Tune")
    model_type = st.selectbox(
        "Model Type:",
        ["isolation_forest", "lof", "autoencoder"],
        index=0,
        format_func=lambda x: {
            "isolation_forest": "Isolation Forest",
            "lof": "Local Outlier Factor",
            "autoencoder": "Autoencoder Neural Network"
        }.get(x, x)
    )
    
    # Create columns for parameter comparison
    st.subheader("Parameter Tuning")
    st.write("Adjust parameters and compare results")
    
    col1, col2 = st.columns(2)
    
    # Configuration 1
    with col1:
        st.markdown("### Configuration 1")
        
        # Common parameters
        contamination1 = st.slider(
            "Contamination (expected % of anomalies):",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.2f",
            key="contamination1"
        )
        
        # Model-specific parameters
        params1 = {"contamination": contamination1, "random_state": 42}
        
        if model_type == "isolation_forest":
            n_estimators1 = st.slider(
                "Number of estimators:",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                key="n_estimators1"
            )
            params1["n_estimators"] = n_estimators1
            
        elif model_type == "lof":
            n_neighbors1 = st.slider(
                "Number of neighbors:",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                key="n_neighbors1"
            )
            params1["n_neighbors"] = n_neighbors1
            
        elif model_type == "autoencoder":
            encoding_dim1 = st.slider(
                "Encoding dimension:",
                min_value=2,
                max_value=20,
                value=8,
                step=2,
                key="encoding_dim1"
            )
            epochs1 = st.slider(
                "Epochs:",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key="epochs1"
            )
            params1["encoding_dim"] = encoding_dim1
            params1["epochs"] = epochs1
            params1["batch_size"] = 32
            params1["dropout_rate"] = 0.2
            params1["hidden_layers"] = [16, 8]
            params1["activation"] = "relu"
            params1["validation_split"] = 0.1
    
    # Configuration 2
    with col2:
        st.markdown("### Configuration 2")
        
        # Common parameters
        contamination2 = st.slider(
            "Contamination (expected % of anomalies):",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            format="%.2f",
            key="contamination2"
        )
        
        # Model-specific parameters
        params2 = {"contamination": contamination2, "random_state": 42}
        
        if model_type == "isolation_forest":
            n_estimators2 = st.slider(
                "Number of estimators:",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                key="n_estimators2"
            )
            params2["n_estimators"] = n_estimators2
            
        elif model_type == "lof":
            n_neighbors2 = st.slider(
                "Number of neighbors:",
                min_value=5,
                max_value=50,
                value=30,
                step=5,
                key="n_neighbors2"
            )
            params2["n_neighbors"] = n_neighbors2
            
        elif model_type == "autoencoder":
            encoding_dim2 = st.slider(
                "Encoding dimension:",
                min_value=2,
                max_value=20,
                value=10,
                step=2,
                key="encoding_dim2"
            )
            epochs2 = st.slider(
                "Epochs:",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
                key="epochs2"
            )
            params2["encoding_dim"] = encoding_dim2
            params2["epochs"] = epochs2
            params2["batch_size"] = 32
            params2["dropout_rate"] = 0.2
            params2["hidden_layers"] = [16, 8]
            params2["activation"] = "relu"
            params2["validation_split"] = 0.1
    
    # Feature selection
    st.subheader("Feature Selection")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date and anomaly columns if they exist
    feature_columns = [col for col in numeric_columns if col not in ['anomaly_score', 'is_anomaly']]
    
    selected_features = st.multiselect(
        "Select features for anomaly detection:",
        options=feature_columns,
        default=feature_columns
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    # Run comparison
    if st.button("Compare Models"):
        with st.spinner(f"Running {model_type.replace('_', ' ').title()} model with different configurations..."):
            try:
                # Create and fit models
                model1 = create_model(model_type, params1)
                model1.fit(df, selected_features)
                result_df1 = model1.predict(df)
                
                model2 = create_model(model_type, params2)
                model2.fit(df, selected_features)
                result_df2 = model2.predict(df)
                
                # Store results in session state
                st.session_state.tuning_results = {
                    "config1": {
                        "params": params1,
                        "results": result_df1,
                        "model": model1
                    },
                    "config2": {
                        "params": params2,
                        "results": result_df2,
                        "model": model2
                    }
                }
                
                st.success("Model comparison completed successfully!")
                
                # Display comparison results
                st.subheader("Comparison Results")
                
                # Summary statistics
                anomaly_count1 = result_df1['is_anomaly'].sum()
                total_count = len(result_df1)
                anomaly_percentage1 = (anomaly_count1 / total_count) * 100
                
                anomaly_count2 = result_df2['is_anomaly'].sum()
                anomaly_percentage2 = (anomaly_count2 / total_count) * 100
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("### Configuration 1 Results")
                    st.write(f"Found {anomaly_count1} anomalies ({anomaly_percentage1:.2f}%)")
                    
                    # Import visualization components
                    from app.ui.components import plot_anomalies
                    
                    # Plot price anomalies
                    if 'Date' in result_df1.columns and 'Close' in result_df1.columns:
                        plot_anomalies(
                            result_df1,
                            x_col='Date',
                            y_col='Close',
                            anomaly_col='is_anomaly',
                            title='Config 1: Price Anomalies'
                        )
                
                with comp_col2:
                    st.markdown("### Configuration 2 Results")
                    st.write(f"Found {anomaly_count2} anomalies ({anomaly_percentage2:.2f}%)")
                    
                    # Plot price anomalies
                    if 'Date' in result_df2.columns and 'Close' in result_df2.columns:
                        plot_anomalies(
                            result_df2,
                            x_col='Date',
                            y_col='Close',
                            anomaly_col='is_anomaly',
                            title='Config 2: Price Anomalies'
                        )
                
                # Apply selected configuration
                st.subheader("Apply Configuration")
                selected_config = st.radio(
                    "Select configuration to apply:",
                    ["Configuration 1", "Configuration 2"]
                )
                
                if st.button("Apply Selected Configuration"):
                    if selected_config == "Configuration 1":
                        st.session_state.anomaly_results = result_df1
                        st.session_state.anomaly_model = model1
                    else:
                        st.session_state.anomaly_results = result_df2
                        st.session_state.anomaly_model = model2
                    
                    st.success(f"Applied {selected_config} successfully!")
                    st.info("You can now view the results in the Anomalies page.")
                
            except Exception as e:
                st.error(f"Error during model comparison: {e}")
    
    # Display previous results if available
    elif 'tuning_results' in st.session_state:
        st.subheader("Previous Comparison Results")
        
        # Get previous results
        result_df1 = st.session_state.tuning_results["config1"]["results"]
        result_df2 = st.session_state.tuning_results["config2"]["results"]
        
        # Summary statistics
        anomaly_count1 = result_df1['is_anomaly'].sum()
        total_count = len(result_df1)
        anomaly_percentage1 = (anomaly_count1 / total_count) * 100
        
        anomaly_count2 = result_df2['is_anomaly'].sum()
        anomaly_percentage2 = (anomaly_count2 / total_count) * 100
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("### Configuration 1 Results")
            st.write(f"Found {anomaly_count1} anomalies ({anomaly_percentage1:.2f}%)")
            
            # Import visualization components
            from app.ui.components import plot_anomalies
            
            # Plot price anomalies
            if 'Date' in result_df1.columns and 'Close' in result_df1.columns:
                plot_anomalies(
                    result_df1,
                    x_col='Date',
                    y_col='Close',
                    anomaly_col='is_anomaly',
                    title='Config 1: Price Anomalies'
                )
        
        with comp_col2:
            st.markdown("### Configuration 2 Results")
            st.write(f"Found {anomaly_count2} anomalies ({anomaly_percentage2:.2f}%)")
            
            # Plot price anomalies
            if 'Date' in result_df2.columns and 'Close' in result_df2.columns:
                plot_anomalies(
                    result_df2,
                    x_col='Date',
                    y_col='Close',
                    anomaly_col='is_anomaly',
                    title='Config 2: Price Anomalies'
                )
        
        # Apply selected configuration
        st.subheader("Apply Configuration")
        selected_config = st.radio(
            "Select configuration to apply:",
            ["Configuration 1", "Configuration 2"]
        )
        
        if st.button("Apply Selected Configuration"):
            if selected_config == "Configuration 1":
                st.session_state.anomaly_results = result_df1
                st.session_state.anomaly_model = st.session_state.tuning_results["config1"]["model"]
            else:
                st.session_state.anomaly_results = result_df2
                st.session_state.anomaly_model = st.session_state.tuning_results["config2"]["model"]
            
            st.success(f"Applied {selected_config} successfully!")
            st.info("You can now view the results in the Anomalies page.")

def render_export_page():
    """
    Render the export page with options to download results.
    """
    st.header("Export Results")
    
    # Check if data is available
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data from the Overview page first.")
        return
    
    # Create tabs for different export options
    tab1, tab2, tab3 = st.tabs(["Cleaned Data", "Anomaly Results", "Model Export"])
    
    # Tab 1: Export cleaned data
    with tab1:
        st.subheader("Export Cleaned Data")
        
        # Get the data
        df = st.session_state.data
        
        # Display data preview
        st.write("Preview:")
        st.dataframe(df.head())
        
        # Export options
        st.write("Export Options:")
        file_format = st.selectbox(
            "Select file format:",
            ["CSV", "Excel", "JSON"],
            key="clean_format"
        )
        
        # Include index option
        include_index = st.checkbox("Include index", value=False, key="clean_index")
        
        # Generate download link
        if st.button("Generate Download Link", key="clean_download"):
            try:
                if file_format == "CSV":
                    csv = df.to_csv(index=include_index)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV File</a>'
                    file_type = "csv"
                    file_name = "cleaned_data.csv"
                elif file_format == "Excel":
                    # Create a BytesIO object
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=include_index, sheet_name='Data')
                    b64 = base64.b64encode(output.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="cleaned_data.xlsx">Download Excel File</a>'
                    file_type = "xlsx"
                    file_name = "cleaned_data.xlsx"
                else:  # JSON
                    json_str = df.to_json(orient='records')
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="cleaned_data.json">Download JSON File</a>'
                    file_type = "json"
                    file_name = "cleaned_data.json"
                
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"Download link generated for {file_name}")
            except Exception as e:
                st.error(f"Error generating download link: {e}")
    
    # Tab 2: Export anomaly results
    with tab2:
        st.subheader("Export Anomaly Results")
        
        if 'anomaly_results' not in st.session_state:
            st.warning("No anomaly detection results available. Please run anomaly detection first.")
        else:
            # Get the anomaly results
            result_df = st.session_state.anomaly_results
            
            # Display results preview
            st.write("Preview:")
            st.dataframe(result_df.head())
            
            # Export options
            st.write("Export Options:")
            
            # File format selection
            file_format = st.selectbox(
                "Select file format:",
                ["CSV", "Excel", "JSON"],
                key="anomaly_format"
            )
            
            # Include only anomalies option
            only_anomalies = st.checkbox("Export only anomalies", value=True, key="only_anomalies")
            
            # Include index option
            include_index = st.checkbox("Include index", value=False, key="anomaly_index")
            
            # Generate download link
            if st.button("Generate Download Link", key="anomaly_download"):
                try:
                    # Filter data if only anomalies are requested
                    export_df = result_df[result_df['is_anomaly'] == 1] if only_anomalies else result_df
                    
                    if file_format == "CSV":
                        csv = export_df.to_csv(index=include_index)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="anomaly_results.csv">Download CSV File</a>'
                        file_type = "csv"
                        file_name = "anomaly_results.csv"
                    elif file_format == "Excel":
                        # Create a BytesIO object
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            export_df.to_excel(writer, index=include_index, sheet_name='Anomalies')
                        b64 = base64.b64encode(output.getvalue()).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="anomaly_results.xlsx">Download Excel File</a>'
                        file_type = "xlsx"
                        file_name = "anomaly_results.xlsx"
                    else:  # JSON
                        json_str = export_df.to_json(orient='records')
                        b64 = base64.b64encode(json_str.encode()).decode()
                        href = f'<a href="data:file/json;base64,{b64}" download="anomaly_results.json">Download JSON File</a>'
                        file_type = "json"
                        file_name = "anomaly_results.json"
                    
                    st.markdown(href, unsafe_allow_html=True)
                    st.success(f"Download link generated for {file_name}")
                except Exception as e:
                    st.error(f"Error generating download link: {e}")
    
    # Tab 3: Export trained model
    with tab3:
        st.subheader("Export Trained Model")
        
        if 'anomaly_model' not in st.session_state:
            st.warning("No trained model available. Please run anomaly detection first.")
        else:
            # Get the model
            model = st.session_state.anomaly_model
            
            # Model info
            st.write("Model Information:")
            model_type = type(model).__name__
            st.write(f"Model Type: {model_type}")
            
            # Model parameters
            st.write("Model Parameters:")
            st.json(model.params)
            
            # Export options
            st.write("Export Options:")
            model_name = st.text_input("Model name:", value="anomaly_model")
            
            # Generate download button
            if st.button("Export Model"):
                try:
                    # Create a temporary directory for model export
                    temp_dir = tempfile.mkdtemp()
                    model_path = os.path.join(temp_dir, model_name)
                    
                    # Save the model
                    model.save(model_path)
                    
                    # Create a zip file
                    zip_path = os.path.join(temp_dir, f"{model_name}.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                if file.startswith(model_name) and not file.endswith('.zip'):
                                    file_path = os.path.join(root, file)
                                    zipf.write(file_path, os.path.basename(file_path))
                    
                    # Read the zip file
                    with open(zip_path, "rb") as f:
                        bytes_data = f.read()
                    
                    # Create download button
                    st.download_button(
                        label="Download Model",
                        data=bytes_data,
                        file_name=f"{model_name}.zip",
                        mime="application/zip"
                    )
                    
                    st.success(f"Model exported successfully as {model_name}.zip")
                except Exception as e:
                    st.error(f"Error exporting model: {e}")

def render_help_page():
    """
    Render the help page with usage instructions.
    """
    st.header("Help")
    st.write("""
    ## How to Use StockPulse
    
    StockPulse is a tool for detecting anomalies in stock market data. Here's how to use it:
    
    1. **Overview Page**: Select a data source (Yahoo Finance, Alpha Vantage, or upload your own CSV) and load the data.
    2. **EDA Page**: Explore the loaded data with visualizations and statistics.
    3. **Anomalies Page**: Run anomaly detection algorithms and view the results.
    4. **Model Tuning Page**: Adjust parameters for the anomaly detection models.
    5. **Export Page**: Download the cleaned data and anomaly reports.
    
    ### Data Sources
    
    - **Yahoo Finance**: Free data source with daily, hourly, and minute-level data.
    - **Alpha Vantage**: Requires an API key, provides daily, weekly, monthly, and intraday data.
    - **Upload CSV**: Upload your own data in CSV or Excel format.
    
    ### Required Columns
    
    The following columns are required for anomaly detection:
    
    - Date: Date or timestamp of the data point
    - Open: Opening price
    - High: Highest price during the period
    - Low: Lowest price during the period
    - Close: Closing price
    
    Optional columns:
    
    - Adj_Close: Adjusted closing price (will be set to Close if missing)
    - Volume: Trading volume
    - Symbol: Ticker symbol
    """)
    
    st.subheader("Popular Stocks")
    stock_options = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms (Facebook)",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "JPM": "JPMorgan Chase & Co."
    }

    # Nicely format stocks
    for ticker, name in stock_options.items():
        st.write(f"- **{name}** ({ticker})")