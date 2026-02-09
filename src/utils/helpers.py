"""
Helper utility functions
"""

import json
import pickle
import joblib
import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from typing import Any, Dict


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Set TensorFlow deterministic operations
    tf.config.experimental.enable_op_determinism()


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save object to pickle file
    
    Args:
        obj: Object to save
        file_path: Path to save pickle file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    """
    Load object from pickle file
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Object loaded from pickle
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)
