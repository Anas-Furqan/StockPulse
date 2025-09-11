from typing import Dict, Any
from .isolation_forest import IsolationForestModel
from .local_outlier_factor import LOFModel
from .autoencoder import AutoencoderModel
from .base_model import BaseAnomalyModel

def create_model(model_type: str, params: Dict[str, Any] = None) -> BaseAnomalyModel:
    """
    Create an anomaly detection model of the specified type.
    
    Args:
        model_type: Type of model to create ('isolation_forest', 'lof', 'autoencoder')
        params: Dictionary of model parameters
        
    Returns:
        Instantiated model
    """
    if model_type == 'isolation_forest':
        return IsolationForestModel(params=params)
    elif model_type == 'lof':
        return LOFModel(params=params)
    elif model_type == 'autoencoder':
        return AutoencoderModel(params=params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")