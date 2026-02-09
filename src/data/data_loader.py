"""
Data loading module for image classification
Handles loading images from directory structure
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


class DataLoader:
    """
    Data loader for image classification datasets
    
    Supports datasets organized in folders:
    data/raw/
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
        test/
            class1/
                img1.jpg
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Path to data directory (should contain raw/ subdirectory)
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_dir}\n"
                f"Please download the dataset and place it in {self.raw_dir}"
            )
    
    def load_train_data(self, validation_split: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data from raw directory
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        train_dir = self.raw_dir / 'train'
        
        if not train_dir.exists():
            # Try alternative structure: all data in one folder
            train_dir = self.raw_dir
        
        # Get class names from subdirectories
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
        if len(class_names) == 0:
            raise ValueError(
                f"No class directories found in {train_dir}\n"
                f"Expected structure: {train_dir}/class1/, {train_dir}/class2/, ..."
            )
        
        print(f"Found {len(class_names)} classes: {class_names[:5]}..." if len(class_names) > 5 else f"Found {len(class_names)} classes: {class_names}")
        
        # Load images and labels
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = train_dir / class_name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                try:
                    # Load image using TensorFlow
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=None)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    continue
        
        if len(images) == 0:
            raise ValueError(f"No images found in {train_dir}")
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        print(f"Loaded {len(X)} training images")
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X[:split_idx]
        X_val = X[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data from raw directory
        
        Returns:
            Tuple of (X_test, y_test)
        """
        test_dir = self.raw_dir / 'test'
        
        if not test_dir.exists():
            print("Warning: Test directory not found. Returning empty arrays.")
            return np.array([]), np.array([])
        
        # Get class names (should match training classes)
        class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
        
        if len(class_names) == 0:
            print("Warning: No test data found.")
            return np.array([]), np.array([])
        
        # Load images and labels
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = test_dir / class_name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=None)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    continue
        
        if len(images) == 0:
            return np.array([]), np.array([])
        
        X_test = np.array(images, dtype=np.float32)
        y_test = np.array(labels, dtype=np.int32)
        
        print(f"Loaded {len(X_test)} test images")
        
        return X_test, y_test
    
    def get_class_names(self) -> List[str]:
        """
        Get list of class names from directory structure
        
        Returns:
            List of class names
        """
        train_dir = self.raw_dir / 'train'
        if not train_dir.exists():
            train_dir = self.raw_dir
        
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        return class_names
    
    def validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate data quality and schema
        
        Args:
            X: Image data array
            y: Label array
            
        Returns:
            True if data is valid
        """
        if len(X) == 0:
            raise ValueError("Data array is empty")
        
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
        
        if X.dtype != np.float32:
            print(f"Warning: X dtype is {X.dtype}, expected float32")
        
        if len(X.shape) < 3:
            raise ValueError(f"Expected image data with shape (N, H, W, C), got {X.shape}")
        
        print(f"Data validation passed: {len(X)} samples, shape {X.shape}")
        return True
