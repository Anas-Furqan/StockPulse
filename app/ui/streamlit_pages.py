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
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import data modules
from app.src.data.fetcher import DataFetcher
from app.src.data.loader import DataLoader
from app.src.data.schema import validate_dataframe, infer_column_mapping, apply_column_mapping
#chat gpt code 

# --- Helper functions: add near top of file (after imports) ---
import pandas as pd

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

def render_overview_page():
    """
    Render the overview page with data source selection and data preview.
    """
    st.header("Overview")
    
    # Create sidebar for data source selection
    st.sidebar.header("Data Source")
    data_source = st.sidebar.selectbox(
        "Select data source:",
        ["Yahoo Finance", "Alpha Vantage", "Upload CSV"]
    )
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if data_source == "Yahoo Finance":
        render_yahoo_finance_input()
    elif data_source == "Alpha Vantage":
        render_alpha_vantage_input()
    elif data_source == "Upload CSV":
        render_csv_upload()
    
    # Display data preview if available
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10))
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(st.session_state.data.describe())
        
        # Display null counts
        st.subheader("Missing Values")
        null_counts = st.session_state.data.isnull().sum()
        st.dataframe(pd.DataFrame({
            'Column': null_counts.index,
            'Missing Values': null_counts.values
        }))
        
        # Display a button to start EDA and anomaly detection
        if st.button("Start EDA & Run Anomaly Detection"):
            st.session_state.run_analysis = True
            st.rerun()

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

def render_csv_upload():
    """
    Render file upload for CSV data source.
    """
    st.subheader("Upload CSV or Excel File")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            file_path = os.path.join(os.getcwd(), "temp_upload.csv")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and validate the file
            df, warnings = data_loader.load_and_validate(file_path)
            
            # Display warnings if any
            if warnings:
                st.warning("\n".join(warnings))
            
            # Infer column mapping
            column_mapping = infer_column_mapping(df)
            
            # Display column mapping for user to adjust
            st.subheader("Column Mapping")
            st.write("Please map your columns to the required format:")
            
            new_mapping = {}
            for canonical_col in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Symbol']:
                default_value = column_mapping.get(canonical_col, "")
                new_mapping[canonical_col] = st.selectbox(
                    f"Map column for {canonical_col}:",
                    ["None"] + list(df.columns),
                    index=0 if default_value == "" else list(df.columns).index(default_value) + 1
                )
            
            # Filter out None values
            new_mapping = {k: v for k, v in new_mapping.items() if v != "None"}
            
            # Options for data processing
            st.subheader("Data Processing Options")
            auto_clean = st.checkbox("Auto-clean & preprocess", value=True)
            auto_predict = st.checkbox("Auto-predict missing columns", value=False)
            
            # Process the data
            if st.button("Process Data"):
                with st.spinner("Processing data..."):
                    try:
                        # Apply column mapping
                        if new_mapping:
                            df_mapped = apply_column_mapping(df, new_mapping)
                        else:
                            df_mapped = apply_column_mapping(df, column_mapping)
                        
                        # Ensure required columns exist
                        required_columns = ['Date', 'Close', 'Volume']
                        missing_columns = [col for col in required_columns if col not in df_mapped.columns]
                
                        if missing_columns:
                            st.warning(f"Missing required columns: {', '.join(missing_columns)}. Some features may not work properly.")
                
                # Store the processed data
                        st.session_state.data = df_mapped
                        st.success("Data processed successfully!")
                
                # Clean up temporary file
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Error processing data: {e}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

def render_eda_page():
    """
    Render the EDA page with data analysis and visualizations.
    """
    st.header("Exploratory Data Analysis")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first from the Overview page.")
        return

        #second start
    # Use a copy of the session data and normalize top-level columns
    df_all = _normalize_df_columns(st.session_state.data.copy())
     # Display basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df_all.describe())

    # Display data types
    st.subheader("Data Types")
    st.dataframe(pd.DataFrame({
        'Column': df_all.dtypes.index,
        'Type': df_all.dtypes.values
    }))

    #second end 
    
    # Display missing values 3rd from column 
    st.subheader("Missing Values")
    missing_data = pd.DataFrame({
        'Column': df_all.isnull().sum().index,
        'Missing Values': df_all.isnull().sum().values,
        'Percentage': (df_all.isnull().sum() / len(df_all) * 100).values
    })
    st.dataframe(missing_data)
    
    # Plot missing values heatmap 4th part
    st.subheader("Missing Values Heatmap")
    fig = px.imshow(
        df_all.isnull(),
        labels=dict(x="Column", y="Row", color="Missing"),
        title="Missing Values Heatmap"
    )
    st.plotly_chart(fig)
    
    # Plot price time series another one 
    st.subheader("Price Time Series")
    if 'Symbol' in df_all.columns:
        symbols = df_all['Symbol'].unique()
        selected_symbol = st.selectbox("Select symbol:", symbols)
        df_symbol = df_all[df_all['Symbol'] == selected_symbol].copy()
    else:
        df_symbol = df_all.copy()
    
    fig = go.Figure()
    
     # Ensure df_symbol columns normalized
    df_symbol = _normalize_df_columns(df_symbol)

    # Detect date/close/volume columns by keyword (robust to variations)
    date_col = _find_col(df_symbol, 'date')
    close_col = _find_col(df_symbol, 'close')
    volume_col = _find_col(df_symbol, 'volume')

    # x values (date or index)
    if date_col:
        x_values = df_symbol[date_col]
    else:
        x_values = df_symbol.index

    # Plot Close time series if found
    fig = go.Figure()
    if close_col is not None:
        fig.add_trace(go.Scatter(x=x_values, y=df_symbol[close_col], mode='lines', name='Close'))
        fig.update_layout(title="Close Price Time Series", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
    else:
        st.warning("No 'Close' column found in the data (searched for columns containing 'close').")

    # Plot volume time series
    if volume_col is not None and not df_symbol[volume_col].isna().all():
        st.subheader("Volume Time Series")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_values, y=df_symbol[volume_col], name='Volume'))
        fig.update_layout(title="Volume Time Series", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig)

    # Plot price distribution (histogram) using found close_col
    st.subheader("Price Distribution")
    if close_col is not None:
        fig = px.histogram(df_symbol, x=close_col, nbins=50, title="Close Price Distribution")
        st.plotly_chart(fig)
    else:
        st.warning("Cannot plot price distribution: 'Close' column not found.")

def render_anomalies_page():
    """
    Render the anomalies page with anomaly detection results.
    """
    st.header("Anomaly Detection")
    
    # Check if data is available
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please load data from the Overview page first.")
        return
    
    # Import models
    from app.src.models import create_model
    
    # Get the data
    df = st.session_state.data
    
    # Model selection
    st.subheader("Select Anomaly Detection Model")
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
    
    # Model parameters
    st.subheader("Model Parameters")
    
    # Common parameters
    col1, col2 = st.columns(2)
    with col1:
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
        with col2:
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
        with col2:
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
        col3, col4 = st.columns(2)
        with col2:
            encoding_dim = st.slider(
                "Encoding dimension:",
                min_value=2,
                max_value=20,
                value=10,
                step=1
            )
        with col3:
            epochs = st.slider(
                "Training epochs:",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )
        with col4:
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
    
    # Run anomaly detection
    if st.button("Run Anomaly Detection"):
        with st.spinner(f"Running {model_type.replace('_', ' ').title()} model..."):
            try:
                # Create and fit the model
                model = create_model(model_type, params)
                model.fit(df, selected_features)
                
                # Predict anomalies
                result_df = model.predict(df)
                
                # Store results in session state
                st.session_state.anomaly_results = result_df
                st.session_state.anomaly_model = model
                
                st.success("Anomaly detection completed successfully!")
                
                # Display results
                st.subheader("Anomaly Detection Results")
                
                # Summary statistics
                anomaly_count = result_df['is_anomaly'].sum()
                total_count = len(result_df)
                anomaly_percentage = (anomaly_count / total_count) * 100
                
                st.write(f"Found {anomaly_count} anomalies out of {total_count} data points ({anomaly_percentage:.2f}%).")
                
                # Display anomalies table
                st.subheader("Anomalies")
                anomalies = result_df[result_df['is_anomaly'] == 1].sort_values(by='anomaly_score', ascending=False)
                st.dataframe(anomalies)
                
                # Visualize anomalies
                st.subheader("Anomaly Visualization")
                
                # Import visualization components
                from app.ui.components import plot_anomalies
                
                # Plot price anomalies
                if 'Date' in result_df.columns and 'Close' in result_df.columns:
                    plot_anomalies(
                        result_df,
                        x_col='Date',
                        y_col='Close',
                        anomaly_col='is_anomaly',
                        title='Price Anomalies'
                    )
                
                # Plot volume anomalies if volume exists
                if 'Date' in result_df.columns and 'Volume' in result_df.columns:
                    plot_anomalies(
                        result_df,
                        x_col='Date',
                        y_col='Volume',
                        anomaly_col='is_anomaly',
                        title='Volume Anomalies'
                    )
                
                # Plot anomaly score distribution
                fig = px.histogram(
                    result_df,
                    x='anomaly_score',
                    color='is_anomaly',
                    nbins=50,
                    title='Anomaly Score Distribution',
                    labels={'is_anomaly': 'Is Anomaly'}
                )
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error during anomaly detection: {e}")
    
    # Display previous results if available
    elif 'anomaly_results' in st.session_state:
        st.subheader("Previous Anomaly Detection Results")
        
        # Summary statistics
        result_df = st.session_state.anomaly_results
        anomaly_count = result_df['is_anomaly'].sum()
        total_count = len(result_df)
        anomaly_percentage = (anomaly_count / total_count) * 100
        
        st.write(f"Found {anomaly_count} anomalies out of {total_count} data points ({anomaly_percentage:.2f}%).")
        
        # Display anomalies table
        st.subheader("Anomalies")
        anomalies = result_df[result_df['is_anomaly'] == 1].sort_values(by='anomaly_score', ascending=False)
        st.dataframe(anomalies)
        
        # Visualize anomalies
        st.subheader("Anomaly Visualization")
        
        # Import visualization components
        from app.ui.components import plot_anomalies
        
        # Plot price anomalies
        if 'Date' in result_df.columns and 'Close' in result_df.columns:
            plot_anomalies(
                result_df,
                x_col='Date',
                y_col='Close',
                anomaly_col='is_anomaly',
                title='Price Anomalies'
            )
        
        # Plot volume anomalies if volume exists
        if 'Date' in result_df.columns and 'Volume' in result_df.columns:
            plot_anomalies(
                result_df,
                x_col='Date',
                y_col='Volume',
                anomaly_col='is_anomaly',
                title='Volume Anomalies'
            )

def render_model_tuning_page():
    """
    Render the model tuning page with parameter settings and comparison.
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