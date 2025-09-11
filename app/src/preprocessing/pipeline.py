import pandas as pd
from typing import Dict, List, Optional, Tuple
from .cleaner import DataCleaner
from .feature_engineering import FeatureEngineer

class PreprocessingPipeline:
    """
    A pipeline for preprocessing stock data, including cleaning and feature engineering.
    """
    
    def __init__(self, fill_method: str = 'ffill'):
        """
        Initialize the PreprocessingPipeline.
        
        Args:
            fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate', 'mean')
        """
        self.cleaner = DataCleaner(fill_method=fill_method)
        self.feature_engineer = FeatureEngineer()
    
    def process(self, df: pd.DataFrame, engineer_features: bool = True) -> pd.DataFrame:
        """
        Process the input DataFrame through the pipeline.
        
        Args:
            df: Input DataFrame
            engineer_features: Whether to perform feature engineering
            
        Returns:
            Processed DataFrame
        """
        # Clean the data
        df_cleaned = self.cleaner.clean(df)
        
        # Engineer features if requested
        if engineer_features:
            df_processed = self.feature_engineer.engineer_features(df_cleaned)
        else:
            df_processed = df_cleaned
        
        return df_processed