"""
Prediction utilities for deployment
Helper functions for making predictions
"""

import numpy as np
from pathlib import Path
from tensorflow import keras
import joblib

from src.data.preprocessor import Preprocessor


def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    """
    Load model and preprocessor
    
    Args:
        model_path: Path to model file
        preprocessor_path: Path to preprocessor file
        
    Returns:
        Tuple of (model, preprocessor)
    """
    model = keras.models.load_model(model_path)
    preprocessor = Preprocessor.load(preprocessor_path)
    return model, preprocessor


def predict_single(model, preprocessor, image: np.ndarray) -> dict:
    """
    Make prediction on a single image
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        image: Image array
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    X_processed, _ = preprocessor.transform(image.reshape((1,) + image.shape), augment=False)
    
    # Predict
    prediction = model.predict(X_processed, verbose=0)
    
    # Format output
    if preprocessor.num_classes == 2:
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(max(prediction[0][0], 1 - prediction[0][0]))
    else:
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': prediction[0].tolist()
    }
