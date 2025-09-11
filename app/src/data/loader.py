import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from .schema import validate_dataframe

class DataLoader:
    """
    A class to load and validate data from various sources.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory to store/load data files
        """
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the loaded data
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise IOError(f"Error loading CSV file: {e}")
    
    def load_excel(self, file_path: str) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame with the loaded data
        """
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            raise IOError(f"Error loading Excel file: {e}")
    
    def save_dataframe(self, df: pd.DataFrame, file_name: str) -> str:
        """
        Save a DataFrame to a CSV file.
        
        Args:
            df: DataFrame to save
            file_name: Name of the file (without extension)
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.data_dir, f"{file_name}.csv")
        df.to_csv(file_path, index=False)
        return file_path
    
    def load_and_validate(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load a file and validate its schema.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (DataFrame, list of validation warnings)
        """
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = self.load_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = self.load_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        
        # Validate the DataFrame
        warnings = validate_dataframe(df)
        
        return df, warnings