import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.src.models.model_factory import create_model
from app.src.models.isolation_forest import IsolationForestModel
from app.src.models.local_outlier_factor import LOFModel
from app.src.models.autoencoder import AutoencoderModel

class TestModelFactory(unittest.TestCase):
    
    def test_create_isolation_forest_model(self):
        model = create_model('isolation_forest', {'contamination': 0.05})
        self.assertIsInstance(model, IsolationForestModel)
        self.assertEqual(model.params['contamination'], 0.05)
    
    def test_create_lof_model(self):
        model = create_model('lof', {'contamination': 0.05})
        self.assertIsInstance(model, LOFModel)
        self.assertEqual(model.params['contamination'], 0.05)
    
    def test_create_autoencoder_model(self):
        model = create_model('autoencoder', {'contamination': 0.05})
        self.assertIsInstance(model, AutoencoderModel)
        self.assertEqual(model.params['contamination'], 0.05)
    
    def test_create_unknown_model(self):
        with self.assertRaises(ValueError):
            create_model('unknown_model', {})

class TestIsolationForestModel(unittest.TestCase):
    
    def setUp(self):
        self.model = IsolationForestModel(params={'contamination': 0.05, 'random_state': 42})
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'Date': pd.date_range(start='2023-01-01', periods=100)
        })
    
    def test_fit_predict(self):
        # Test fit method
        self.model.fit(self.df, ['feature1', 'feature2'])
        self.assertIsNotNone(self.model.model)
        
        # Test predict method
        result_df = self.model.predict(self.df)
        self.assertIn('anomaly_score', result_df.columns)
        self.assertIn('is_anomaly', result_df.columns)
        
        # Check if anomaly percentage matches contamination
        anomaly_percentage = result_df['is_anomaly'].mean()
        self.assertAlmostEqual(anomaly_percentage, 0.05, delta=0.02)

class TestLOFModel(unittest.TestCase):
    
    def setUp(self):
        self.model = LOFModel(params={'contamination': 0.05, 'n_neighbors': 20})
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'Date': pd.date_range(start='2023-01-01', periods=100)
        })
    
    def test_fit_predict(self):
        # Test fit method
        self.model.fit(self.df, ['feature1', 'feature2'])
        self.assertIsNotNone(self.model.model)
        
        # Test predict method
        result_df = self.model.predict(self.df)
        self.assertIn('anomaly_score', result_df.columns)
        self.assertIn('is_anomaly', result_df.columns)

if __name__ == '__main__':
    unittest.main()