import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class DataCleaner:
    """
    A class to clean and preprocess stock data.
    """
    
    def __init__(self, fill_method: str = 'ffill'):
        """
        Initialize the DataCleaner.
        
        Args:
            fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate', 'mean')
        """
        self.fill_method = fill_method
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame by handling missing values, outliers, and duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Sort by Date and Symbol (if available)
        if 'Symbol' in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values(['Symbol', 'Date'])
        else:
            df_cleaned = df_cleaned.sort_values('Date')
        
        # Handle duplicates
        df_cleaned = self._remove_duplicates(df_cleaned)
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Handle outliers
        df_cleaned = self._handle_outliers(df_cleaned)
        
        return df_cleaned
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows based on Date and Symbol.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        if 'Symbol' in df.columns:
            return df.drop_duplicates(subset=['Date', 'Symbol'], keep='first')
        else:
            return df.drop_duplicates(subset=['Date'], keep='first')
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # Group by Symbol if available
        if 'Symbol' in df.columns:
            result = pd.DataFrame()
            for symbol, group in df.groupby('Symbol'):
                result = pd.concat([result, self._fill_missing_values(group)])
            return result
        else:
            return self._fill_missing_values(df)
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using the specified method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values filled
        """
        # Make a copy
        df_filled = df.copy()
        
        # Fill missing values in numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in numeric_cols:
            if col in df_filled.columns:
                if self.fill_method == 'ffill':
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                    # If there are still NaNs at the beginning, fill them with bfill
                    df_filled[col] = df_filled[col].fillna(method='bfill')
                elif self.fill_method == 'bfill':
                    df_filled[col] = df_filled[col].fillna(method='bfill')
                    # If there are still NaNs at the end, fill them with ffill
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                elif self.fill_method == 'interpolate':
                    df_filled[col] = df_filled[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                elif self.fill_method == 'mean':
                    mean_val = df_filled[col].mean()
                    df_filled[col] = df_filled[col].fillna(mean_val)
        
        return df_filled
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using IQR method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        # Make a copy
        df_no_outliers = df.copy()
        
        # Handle outliers in numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        
        # Group by Symbol if available
        if 'Symbol' in df_no_outliers.columns:
            for symbol, group in df_no_outliers.groupby('Symbol'):
                for col in numeric_cols:
                    if col in group.columns:
                        # Calculate IQR
                        Q1 = group[col].quantile(0.25)
                        Q3 = group[col].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        # Define bounds
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Cap outliers instead of removing them
                        mask = (df_no_outliers['Symbol'] == symbol)
                        df_no_outliers.loc[mask & (df_no_outliers[col] < lower_bound), col] = lower_bound
                        df_no_outliers.loc[mask & (df_no_outliers[col] > upper_bound), col] = upper_bound
        else:
            for col in numeric_cols:
                if col in df_no_outliers.columns:
                    # Calculate IQR
                    Q1 = df_no_outliers[col].quantile(0.25)
                    Q3 = df_no_outliers[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df_no_outliers.loc[df_no_outliers[col] < lower_bound, col] = lower_bound
                    df_no_outliers.loc[df_no_outliers[col] > upper_bound, col] = upper_bound
        
        return df_no_outliers