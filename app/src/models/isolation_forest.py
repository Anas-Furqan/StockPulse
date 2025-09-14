import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from .base_model import BaseAnomalyModel

class IsolationForestModel(BaseAnomalyModel):
    """
    Anomaly detection model using Isolation Forest algorithm.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Isolation Forest model.
        
        Args:
            params: Dictionary of model parameters
        """
        default_params = {
            'n_estimators': 100,
            'max_samples': 'auto',
            'contamination': 'auto',
            'random_state': 42
        }
        
        # Update default parameters with provided parameters
        if params:
            default_params.update(params)
        
        super().__init__(name="isolation_forest", params=default_params)
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None) -> 'IsolationForestModel':
        """
        Fit the Isolation Forest model to the data.
        
        Args:
            df: Input DataFrame
            feature_columns: List of columns to use as features
            
        Returns:
            Self for method chaining
        """
        # Use all numeric columns if feature_columns is not provided
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Validate data and feature columns
        if len(df) == 0:
            raise ValueError("DataFrame is empty. Cannot fit model on empty data.")
            
        if len(feature_columns) == 0:
            raise ValueError("No feature columns provided or found. Cannot fit model without features.")
            
        # Check if feature columns exist in the DataFrame
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns {missing_cols} not found in DataFrame.")
            
        # Check if we have enough data after removing NaN values
        df_features = df[feature_columns].dropna()
        if len(df_features) == 0:
            raise ValueError("All rows contain NaN values for the selected features.")
        
        # Store feature columns for prediction
        self.feature_columns = feature_columns
        
        # Create and fit the model
        self.model = SklearnIsolationForest(**self.params)
        self.model.fit(df_features)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with anomaly scores and labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Validate input data
        if len(df) == 0:
            raise ValueError("Cannot predict on empty DataFrame")
            
        # Check if feature columns exist in the DataFrame
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns {missing_cols} not found in DataFrame")
        
        # Make a copy of the input DataFrame
        result_df = df.copy()
        
        # Handle missing values by filling with mean or 0
        df_features = df[self.feature_columns].copy()
        for col in self.feature_columns:
            if df_features[col].isna().any():
                # Fill with mean if possible, otherwise use 0
                if not df_features[col].isna().all():
                    df_features[col] = df_features[col].fillna(df_features[col].mean())
                else:
                    df_features[col] = df_features[col].fillna(0)
        
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        raw_scores = self.model.predict(df_features)
        
        # Convert to anomaly scores (higher means more anomalous)
        # Isolation Forest returns -1 for anomalies and 1 for normal points
        # We convert to a 0-1 scale where 1 is anomalous
        anomaly_scores = np.where(raw_scores == -1, 1, 0)
        
        # Get decision function scores (negative means more anomalous)
        decision_scores = self.model.decision_function(df_features)
        
        # Convert to anomaly scores (higher means more anomalous)
        # Handle the case where all scores are the same
        score_range = np.max(decision_scores) - np.min(decision_scores)
        if score_range > 0:
            normalized_scores = 1 - (decision_scores - np.min(decision_scores)) / score_range
        else:
            normalized_scores = np.zeros_like(decision_scores)
        
        # Add results to DataFrame
        result_df['anomaly_score'] = normalized_scores
        result_df['is_anomaly'] = anomaly_scores
        
        return result_df