"""
Flask REST API for model inference
Provides endpoints for health check, model info, and predictions
"""

import os
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import joblib

from src.data.preprocessor import Preprocessor

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and preprocessor
MODEL = None
PREPROCESSOR = None
MODEL_INFO = None


def load_model_and_preprocessor():
    """Load model and preprocessor at startup"""
    global MODEL, PREPROCESSOR, MODEL_INFO
    
    # Paths (inside Docker container)
    model_path = Path('/app/models/best_model.h5')
    preprocessor_path = Path('/app/models/preprocessor.pkl')
    metrics_path = Path('/app/models/metrics.json')
    
    # Fallback to relative paths for local testing
    if not model_path.exists():
        model_path = Path('models/best_model.h5')
    if not preprocessor_path.exists():
        preprocessor_path = Path('models/preprocessor.pkl')
    if not metrics_path.exists():
        metrics_path = Path('models/metrics.json')
    
    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    MODEL = keras.models.load_model(str(model_path))
    print("✓ Model loaded successfully")
    
    # Load preprocessor
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    print(f"Loading preprocessor from {preprocessor_path}...")
    PREPROCESSOR = Preprocessor.load(str(preprocessor_path))
    print("✓ Preprocessor loaded successfully")
    
    # Load model info
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            MODEL_INFO = json.load(f)
    else:
        MODEL_INFO = {
            'input_shape': str(MODEL.input_shape),
            'output_shape': str(MODEL.output_shape)
        }
    
    print("API initialization complete")


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    
    Returns:
        JSON with API health status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'preprocessor_loaded': PREPROCESSOR is not None,
        'version': '1.0.0'
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get model information
    
    Returns:
        JSON with model architecture and metrics
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'architecture': 'Deep Neural Network (CNN)',
        'framework': 'TensorFlow/Keras',
        'input_shape': str(MODEL.input_shape),
        'output_shape': str(MODEL.output_shape),
        'num_trainable_params': int(sum([tf.size(w).numpy() for w in MODEL.trainable_variables])),
        'metrics': MODEL_INFO.get('best_val_accuracy', None),
        'preprocessor': {
            'target_size': PREPROCESSOR.target_size if PREPROCESSOR else None,
            'normalize': PREPROCESSOR.normalize if PREPROCESSOR else None,
            'num_classes': PREPROCESSOR.num_classes if PREPROCESSOR else None
        }
    }
    
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions on input data
    
    Expected JSON format:
    {
        "data": [[pixel values...]] or base64 encoded image
    }
    
    For image classification, expects:
    - List of images as arrays (flattened or 2D/3D)
    - Each image should match the expected input shape
    
    Returns:
        JSON with predictions, confidence scores, and probabilities
    """
    if MODEL is None or PREPROCESSOR is None:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500
    
    try:
        # Parse input
        input_data = request.json.get('data')
        
        if input_data is None:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to numpy array
        if isinstance(input_data, list):
            input_array = np.array(input_data, dtype=np.float32)
        else:
            return jsonify({'error': 'Invalid data format. Expected list of arrays.'}), 400
        
        # Handle different input shapes
        # If flattened, reshape based on expected input shape
        expected_shape = MODEL.input_shape[1:]  # Remove batch dimension
        
        if len(input_array.shape) == 1:
            # Single flattened image
            if input_array.shape[0] == np.prod(expected_shape):
                input_array = input_array.reshape((1,) + expected_shape)
            else:
                return jsonify({
                    'error': f'Input shape mismatch. Expected {np.prod(expected_shape)} values, got {input_array.shape[0]}'
                }), 400
        elif len(input_array.shape) == 2:
            # Multiple flattened images or single 2D image
            if input_array.shape[1] == np.prod(expected_shape):
                # Multiple flattened images
                input_array = input_array.reshape((input_array.shape[0],) + expected_shape)
            elif len(expected_shape) == 2 and input_array.shape == expected_shape:
                # Single 2D image (grayscale)
                input_array = input_array.reshape((1,) + expected_shape)
            else:
                return jsonify({'error': 'Input shape mismatch'}), 400
        elif len(input_array.shape) == 3:
            # Single 3D image (H, W, C)
            if input_array.shape == expected_shape:
                input_array = input_array.reshape((1,) + expected_shape)
            else:
                return jsonify({'error': 'Input shape mismatch'}), 400
        elif len(input_array.shape) == 4:
            # Batch of images (N, H, W, C)
            if input_array.shape[1:] != expected_shape:
                return jsonify({'error': 'Input shape mismatch'}), 400
        else:
            return jsonify({'error': 'Invalid input dimensions'}), 400
        
        # Ensure pixel values are in [0, 255] range if needed
        if input_array.max() > 1.0:
            input_array = input_array / 255.0
        
        # Preprocess
        X_processed, _ = PREPROCESSOR.transform(input_array, augment=False)
        
        # Make predictions
        predictions = MODEL.predict(X_processed, verbose=0)
        
        # Format output
        if PREPROCESSOR.num_classes == 2:
            # Binary classification
            predicted_classes = (predictions > 0.5).astype(int).flatten()
            confidence_scores = np.maximum(predictions, 1 - predictions).flatten()
        else:
            # Multi-class classification
            predicted_classes = np.argmax(predictions, axis=1)
            confidence_scores = np.max(predictions, axis=1)
        
        # Prepare response
        response = {
            'predictions': predicted_classes.tolist(),
            'confidence': confidence_scores.tolist(),
            'probabilities': predictions.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions (alternative endpoint)
    
    Same as /predict but explicitly for batch processing
    """
    return predict()


if __name__ == '__main__':
    # Load model and preprocessor at startup
    try:
        load_model_and_preprocessor()
    except Exception as e:
        print(f"Error loading model/preprocessor: {e}")
        print("API will start but predictions will fail until model is loaded")
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
