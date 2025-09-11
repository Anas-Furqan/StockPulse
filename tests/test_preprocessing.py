import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.src.preprocessing.feature_engineering import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=100),
            'Open': np.random.rand(100) * 100 + 100,
            'High': np.random.rand(100) * 100 + 110,
            'Low': np.random.rand(100) * 100 + 90,
            'Close': np.random.rand(100) * 100 + 105,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        self.feature_engineer = FeatureEngineer()
    
    def test_add_technical_indicators(self):
        # Add technical indicators
        result_df = self.feature_engineer.add_technical_indicators(self.df)
        
        # Check if new columns were added
        self.assertIn('MA_5', result_df.columns)
        self.assertIn('MA_20', result_df.columns)
        self.assertIn('RSI', result_df.columns)
        self.assertIn('MACD', result_df.columns)
    
    def test_add_date_features(self):
        # Add date features
        result_df = self.feature_engineer.add_date_features(self.df)
        
        # Check if new columns were added
        self.assertIn('day_of_week', result_df.columns)
        self.assertIn('month', result_df.columns)
        self.assertIn('quarter', result_df.columns)

if __name__ == '__main__':
    unittest.main()