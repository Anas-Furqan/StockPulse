import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.neighbors import LocalOutlierFactor
from .base_model import BaseAnomalyModel

class LOFModel(BaseAnomalyModel):
    """
    Anomaly detection model using Local Outlier Factor algorithm.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the LOF model.
        
        Args:
            params: Dictionary of model parameters
        """
        default_params = {
            'n_neighbors': 20,
            'contamination': 'auto',
            'novelty': False
        }
        
        # Update default parameters with provided parameters
        if params:
            default_params.update(params)
        
        super().__init__(name="local_outlier_factor", params=default_params)
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None) -> 'LOFModel':
        """
        Fit the LOF model to the data.
        
        Args:
            df: Input DataFrame
            feature_columns: List of columns to use as features
            
        Returns:
            Self for method chaining
        """
        # Use all numeric columns if feature_columns is not provided
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Store feature columns for prediction
        self.feature_columns = feature_columns
        
        # Create and fit the model
        self.model = LocalOutlierFactor(**self.params)
        
        # LOF with novelty=False doesn't have a separate fit method
        # It uses fit_predict directly
        if not self.params.get('novelty', False):
            self._fit_result = self.model.fit_predict(df[feature_columns])
        else:
            self.model.fit(df[feature_columns])
        
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
        
        # Make a copy of the input DataFrame
        result_df = df.copy()
        
        # Get anomaly scores
        if self.params.get('novelty', False):
            # For novelty=True, we can use predict and decision_function
            raw_scores = self.model.predict(df[self.feature_columns])
            decision_scores = self.model.decision_function(df[self.feature_columns])
        else:
            # For novelty=False, we use the stored fit_predict result for training data
            # For new data, we need to create a new LOF instance
            if len(df) == len(self._fit_result):
                raw_scores = self._fit_result
                # Get negative outlier factor (higher means more anomalous)
                decision_scores = -self.model.negative_outlier_factor_
            else:
                # For new data, create a new LOF instance with novelty=True
                new_model = LocalOutlierFactor(**{**self.params, 'novelty': True})
                new_model.fit(df[self.feature_columns])
                raw_scores = new_model.predict(df[self.feature_columns])
                decision_scores = new_model.decision_function(df[self.feature_columns])
        
        # Convert to anomaly scores (higher means more anomalous)
        # LOF returns -1 for anomalies and 1 for normal points
        anomaly_scores = np.where(raw_scores == -1, 1, 0)
        
        # Normalize decision scores to 0-1 range (higher means more anomalous)
        normalized_scores = (decision_scores - np.min(decision_scores)) / (np.max(decision_scores) - np.min(decision_scores))
        
        # Add results to DataFrame
        result_df['anomaly_score'] = normalized_scores
        result_df['is_anomaly'] = anomaly_scores
        
        return result_df