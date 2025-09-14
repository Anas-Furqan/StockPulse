import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import joblib
import os

class BaseAnomalyModel:
    """
    Base class for anomaly detection models.
    """ 
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize the base anomaly detection model.
        
        Args:
            name: Name of the model
            params: Dictionary of model parameters
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.feature_columns = None
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None) -> 'BaseAnomalyModel':
        """
        Fit the model to the data.
        
        Args:
            df: Input DataFrame
            feature_columns: List of columns to use as features
            
        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with anomaly scores and labels
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def save(self, model_dir: str) -> str:
        """
        Save the model to disk.
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.name}.joblib")
        
        # Save model and metadata
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'params': self.params
        }, model_path)
        
        return model_path
    
    def load(self, model_path: str) -> 'BaseAnomalyModel':
        """
        Load the model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model and metadata
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.params.update(data['params'])
        
        return self