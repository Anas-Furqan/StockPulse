import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import joblib
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from .base_model import BaseAnomalyModel

class AutoencoderModel(BaseAnomalyModel):
    """
    Anomaly detection model using Autoencoder neural network.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Autoencoder model.
        
        Args:
            params: Dictionary of model parameters
        """
        default_params = {
            'encoding_dim': 10,
            'hidden_layers': [20, 10],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.1,
            'contamination': 0.05,
            'random_state': 42
        }
        
        # Update default parameters with provided parameters
        if params:
            default_params.update(params)
        
        super().__init__(name="autoencoder", params=default_params)
        self.scaler = StandardScaler()
        self.threshold = None
    
    def _build_model(self, input_dim: int) -> Model:
        """
        Build the autoencoder model architecture.
        
        Args:
            input_dim: Dimension of the input data
            
        Returns:
            Compiled Keras model
        """
        # Set random seed for reproducibility
        tf.random.set_seed(self.params['random_state'])
        np.random.seed(self.params['random_state'])
        
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoder = input_layer
        for units in self.params['hidden_layers']:
            encoder = Dense(units, activation=self.params['activation'])(encoder)
            encoder = Dropout(self.params['dropout_rate'])(encoder)
        
        # Bottleneck layer
        bottleneck = Dense(self.params['encoding_dim'], activation=self.params['activation'])(encoder)
        
        # Decoder
        decoder = bottleneck
        for units in reversed(self.params['hidden_layers']):
            decoder = Dense(units, activation=self.params['activation'])(decoder)
            decoder = Dropout(self.params['dropout_rate'])(decoder)
        
        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoder)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None) -> 'AutoencoderModel':
        """
        Fit the Autoencoder model to the data.
        
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
        
        # Scale the data
        X = self.scaler.fit_transform(df[feature_columns])
        
        # Build the model
        self.model = self._build_model(input_dim=len(feature_columns))
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        self.model.fit(
            X, X,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=self.params['validation_split'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate reconstruction errors
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Set threshold based on contamination parameter
        self.threshold = np.percentile(mse, 100 * (1 - self.params['contamination']))
        
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
        
        # Scale the data
        X = self.scaler.transform(df[self.feature_columns])
        
        # Calculate reconstruction errors
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Normalize scores to 0-1 range
        min_mse = np.min(mse)
        max_mse = np.max(mse)
        normalized_scores = (mse - min_mse) / (max_mse - min_mse)
        
        # Determine anomalies based on threshold
        anomaly_labels = np.where(mse > self.threshold, 1, 0)
        
        # Add results to DataFrame
        result_df['anomaly_score'] = normalized_scores
        result_df['is_anomaly'] = anomaly_labels
        
        return result_df
    
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
        model_path = os.path.join(model_dir, f"{self.name}")
        
        # Save Keras model
        self.model.save(f"{model_path}_keras.h5")
        
        # Save other components
        joblib.dump({
            'feature_columns': self.feature_columns,
            'params': self.params,
            'scaler': self.scaler,
            'threshold': self.threshold
        }, f"{model_path}_components.joblib")
        
        return model_path
    
    def load(self, model_path: str) -> 'AutoencoderModel':
        """
        Load the model from disk.
        
        Args:
            model_path: Path to the saved model (without extension)
            
        Returns:
            Self for method chaining
        """
        keras_path = f"{model_path}_keras.h5"
        components_path = f"{model_path}_components.joblib"
        
        if not os.path.exists(keras_path) or not os.path.exists(components_path):
            raise FileNotFoundError(f"Model files not found: {keras_path} or {components_path}")
        
        # Load Keras model
        self.model = load_model(keras_path)
        
        # Load other components
        components = joblib.load(components_path)
        self.feature_columns = components['feature_columns']
        self.params.update(components['params'])
        self.scaler = components['scaler']
        self.threshold = components['threshold']
        
        return self