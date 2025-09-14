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

    if uploaded_file is not None:
        file_path = None
        try:
            # Write uploaded file to temp location
            file_extension = uploaded_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                file_path = temp_file.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            return

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
                    try:
                        # Check required columns
                        required = ['Date', 'Close']
                        missing = [col for col in required if col not in new_mapping]
                        if missing:
                            st.error(f"Missing required columns: {', '.join(missing)}. Please map all required columns.")
                            return

                        # Create a copy of the dataframe and apply mapping
                        mapped_df = df.copy()
                        mapped_df = mapped_df.rename(columns={v: k for k, v in new_mapping.items()})

                        # Validate required columns exist after mapping
                        if not all(col in mapped_df.columns for col in ['Date', 'Close']):
                            st.error("Required columns missing after mapping. Please check your column selections.")
                            return

                        # First pass: Convert date and validate
                        mapped_df['Date'] = pd.to_datetime(mapped_df['Date'], errors='coerce')
                        mapped_df = mapped_df.dropna(subset=['Date'])

                        if mapped_df.empty:
                            st.error("No valid dates found in the Date column. Please check your data.")
                            return

                        # Second pass: Convert numeric columns
                        numeric_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
                        for col in numeric_cols:
                            if col in mapped_df.columns:
                                mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')

                        # Ensure we have valid Close prices
                        mapped_df = mapped_df.dropna(subset=['Close'])

                        if mapped_df.empty:
                            st.error("No valid numeric data found in the Close column. Please check your data.")
                            return
                            
                    except Exception as e:
                        st.error(f"Error during data processing: {str(e)}")
                        return

                    # Ensure Date column is datetime
                    mapped_df['Date'] = pd.to_datetime(mapped_df['Date'], errors='coerce')

                    # Convert numeric columns
                    for col in ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']:
                        if col in mapped_df.columns:
                            mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')

                    # Clean and process the data
                    if auto_clean:
                        # Fill missing values in other numeric columns
                        for col in mapped_df.select_dtypes(include=[np.number]).columns:
                            if col != 'Close':  # We already handled Close
                                mapped_df[col] = mapped_df[col].fillna(method='ffill').fillna(method='bfill')
                        
                        # Add Adj_Close if missing
                        if 'Adj_Close' not in mapped_df.columns and 'Close' in mapped_df.columns:
                            mapped_df['Adj_Close'] = mapped_df['Close']
                        
                        # Ensure symbol exists
                        if 'Symbol' not in mapped_df.columns:
                            mapped_df['Symbol'] = uploaded_file.name.split('.')[0]
                    
                    # Sort by date
                    mapped_df = mapped_df.sort_values('Date').reset_index(drop=True)
                    
                    # Display data quality metrics
                    st.write("Data Quality Check:")
                    quality_col1, quality_col2 = st.columns(2)
                    
                    with quality_col1:
                        st.metric("Total Rows", len(mapped_df))
                        st.metric("Date Range", f"{mapped_df['Date'].min().date()} to {mapped_df['Date'].max().date()}")
                    
                    with quality_col2:
                        null_counts = mapped_df[['Close']].isnull().sum()
                        st.metric("Missing Close Values", int(null_counts['Close']))
                        if 'Volume' in mapped_df.columns:
                            null_volume = mapped_df['Volume'].isnull().sum()
                            st.metric("Missing Volume Values", int(null_volume))

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
                            
                        except ImportError as e:
                            st.error(f"Could not import required model: {str(e)}")
                            return
                        except Exception as e:
                            st.error(f"Error preparing data for anomaly detection: {str(e)}")
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
                    except Exception:
                        pass

        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
    else:
        st.info("Please upload a file using the uploader above.")