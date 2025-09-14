"""
StockPulse Streamlit UI components.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Add the app directory to the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import data modules
from app.src.data.fetcher import DataFetcher
from app.src.data.loader import DataLoader
from app.src.data.schema import validate_dataframe, infer_column_mapping, apply_column_mapping

# Initialize data fetcher and loader
data_fetcher = DataFetcher()
data_loader = DataLoader()

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

def auto_fetch_default_stock():
    """
    Automatically fetch data for the default stock symbol.
    """
    try:
        # Get default stock symbol from session state or fallback
        symbol = st.session_state.get('default_stock', 'AAPL')
        # Set default date range (1 year)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        # Fetch data
        data_dict = data_fetcher.fetch_from_yahoo(
            tickers=[symbol],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
        # Process data
        if data_dict:
            frames = []
            for sym, df in data_dict.items():
                df_norm = _normalize_df_columns(df, symbol_hint=sym)
                frames.append(df_norm)
            st.session_state.data = pd.concat(frames, ignore_index=True, sort=False)
    except Exception as e:
        st.error(f"Error auto-fetching data: {e}")
        st.session_state.data = None

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
                    date_range = "Invalid dates"
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
        st.dataframe(df.head(10))

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
                    frames = []
                    for sym, df in data_dict.items():
                        df_norm = _normalize_df_columns(df, symbol_hint=sym)
                        frames.append(df_norm)
                    st.session_state.data = pd.concat(frames, ignore_index=True, sort=False)
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
        help="Upload your stock data file in CSV, JSON, or XML format"
    )

    if uploaded_file is not None:
        try:
            # Write uploaded file to temp location
            file_extension = uploaded_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                file_path = temp_file.name

            # Load data based on file type
            if file_extension == 'csv':
                df, warnings = data_loader.load_and_validate(file_path)
            elif file_extension == 'json':
                try:
                    df = pd.read_json(file_path)
                    warnings = []
                except Exception:
                    df = parse_json_file(file_path)
                    warnings = []
            elif file_extension == 'xml':
                df = parse_xml_file(file_path)
                warnings = []
            else:
                df, warnings = data_loader.load_and_validate(file_path)

            if df is None or df.empty:
                st.error("Uploaded file is empty or could not be loaded. Please check your file.")
                return

            st.info(f"Uploaded file: {uploaded_file.name} ({file_extension.upper()}) - {len(df)} rows, {len(df.columns)} columns")
            if warnings:
                st.warning("\n".join(warnings))

            # Infer column mapping
            column_mapping = infer_column_mapping(df)

            st.subheader("Column Mapping")
            st.write("Please map your columns to the required format:")
            new_mapping = {}

            # Create two columns for better layout
            col1, col2 = st.columns(2)

            # Essential columns in first column
            with col1:
                for canonical_col in ['Date', 'Close', 'Volume', 'Symbol']:
                    options = ["None"] + list(df.columns)
                    default = column_mapping.get(canonical_col, "None")
                    new_mapping[canonical_col] = st.selectbox(
                        f"{canonical_col} column", 
                        options, 
                        index=options.index(default) if default in options else 0,
                        key=f"map_{canonical_col}"
                    )

            # Optional columns in second column
            with col2:
                for canonical_col in ['Open', 'High', 'Low', 'Adj_Close']:
                    options = ["None"] + list(df.columns)
                    default = column_mapping.get(canonical_col, "None")
                    new_mapping[canonical_col] = st.selectbox(
                        f"{canonical_col} column", 
                        options, 
                        index=options.index(default) if default in options else 0,
                        key=f"map_{canonical_col}"
                    )

            # Filter out None values
            new_mapping = {k: v for k, v in new_mapping.items() if v != "None"}

            # Display the current mapping
            if new_mapping:
                st.info("Current mapping: " + ", ".join([f"{k} → {v}" for k, v in new_mapping.items()]))
            else:
                st.warning("No columns mapped yet. Please select at least Date and Close columns.")

            st.subheader("Data Processing Options")
            auto_clean = st.checkbox("Auto-clean & preprocess", value=True)
            auto_predict = st.checkbox("Auto-detect anomalies", value=True)

            if st.button("Process Data"):
                with st.spinner("Processing data..."):
                    # Check required columns
                    required = ['Date', 'Close']
                    missing = [col for col in required if col not in new_mapping]
                    if missing:
                        st.error(f"Missing required columns: {', '.join(missing)}. Please map all required columns.")
                        return

                    # Apply mapping
                    mapped_df = apply_column_mapping(df, new_mapping)
                    if mapped_df is None or mapped_df.empty:
                        st.error("No data left after mapping and cleaning. Please check your mapping and file contents.")
                        return

                    # Ensure Date column is datetime
                    mapped_df['Date'] = pd.to_datetime(mapped_df['Date'], errors='coerce')

                    # Convert numeric columns
                    for col in ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']:
                        if col in mapped_df.columns:
                            mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')

                    # Drop rows with missing required values
                    mapped_df = mapped_df.dropna(subset=['Date', 'Close'])
                    
                    if mapped_df.empty:
                        st.error("No valid data left after cleaning. Please check your mapping and file contents.")
                        return

                    # Sort by date
                    mapped_df = mapped_df.sort_values('Date').reset_index(drop=True)

                    # Store in session state
                    st.session_state.data = mapped_df
                    st.session_state.data_source = "file_upload"
                    st.success(f"Data processed successfully! {len(mapped_df)} rows ready for analysis.")

                    # Run automatic anomaly detection if selected
                    if auto_predict:
                        try:
                            from app.src.models import create_model
                            
                            # Select numeric columns for features
                            numeric_cols = mapped_df.select_dtypes(include=[np.number]).columns.tolist()
                            numeric_cols = [col for col in numeric_cols if col not in ['anomaly_score', 'is_anomaly']]
                            
                            if len(numeric_cols) == 0:
                                st.warning("No numeric columns available for anomaly detection.")
                                return
                            
                            # Create and fit isolation forest model
                            model = create_model('isolation_forest', {
                                'contamination': 0.1,
                                'n_estimators': 100,
                                'random_state': 42
                            })
                            
                            # Fit model and get anomaly scores
                            feature_data = mapped_df[numeric_cols]
                            feature_data = feature_data.fillna(feature_data.mean())
                            model.fit(feature_data)
                            anomaly_scores = model.decision_scores_
                            anomaly_labels = model.labels_
                            
                            # Add results to dataframe
                            result_df = mapped_df.copy()
                            result_df['anomaly_score'] = anomaly_scores
                            result_df['is_anomaly'] = anomaly_labels
                            
                            # Store in session state
                            st.session_state.anomaly_results = result_df
                            st.session_state.anomaly_model = model
                            st.session_state.anomaly_model_type = 'isolation_forest'
                            st.session_state.anomaly_features = numeric_cols
                            
                            # Show anomaly count
                            anomaly_count = anomaly_labels.sum()
                            st.success(f"Detected {anomaly_count} potential anomalies!")
                            
                        except Exception as e:
                            st.warning(f"Could not run anomaly detection: {e}")

                    # Show data preview with tabs
                    preview_tabs = st.tabs(["Data Preview", "Statistics", "Column Info"])
                    
                    with preview_tabs[0]:
                        st.dataframe(mapped_df.head(20))
                    
                    with preview_tabs[1]:
                        st.write("Basic Statistics")
                        st.dataframe(mapped_df.describe())
                    
                    with preview_tabs[2]:
                        st.write("Column Information")
                        col_info = pd.DataFrame({
                            'Column': mapped_df.columns,
                            'Type': mapped_df.dtypes,
                            'Non-Null Count': mapped_df.count(),
                            'Null Count': mapped_df.isna().sum(),
                            'Null %': (mapped_df.isna().sum() / len(mapped_df) * 100).round(2)
                        })
                        st.dataframe(col_info)
                    
                    # Clean up
                    try:
                        os.remove(file_path)
                    except:
                        pass

        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
    else:
        st.info("Please upload a file using the uploader above.")
