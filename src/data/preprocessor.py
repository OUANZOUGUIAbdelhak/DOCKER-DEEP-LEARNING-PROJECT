"""
Data preprocessing module for image classification
Handles image preprocessing, augmentation, and normalization
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Preprocessor:
    """
    Preprocessor for image classification tasks
    
    Features:
    - Image resizing
    - Normalization (pixel values to [0, 1])
    - Data augmentation for training
    - One-hot encoding for labels
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augmentation: bool = True
    ):
        """
        Initialize Preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
            augmentation: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augmentation = augmentation
        self.num_classes = None
        self.is_fitted = False
        
        # Setup augmentation generator
        if self.augmentation:
            self.augmentation_generator = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            self.augmentation_generator = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Preprocessor':
        """
        Fit preprocessor on training data
        
        Args:
            X: Training images (N, H, W, C)
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        # Determine number of classes
        self.num_classes = len(np.unique(y))
        self.is_fitted = True
        
        print(f"Preprocessor fitted: {self.num_classes} classes, target size {self.target_size}")
        return self
    
    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augment: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform images and labels
        
        Args:
            X: Image array (N, H, W, C)
            y: Label array (optional)
            augment: Whether to apply augmentation (only for training)
            
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        if not self.is_fitted and y is not None:
            self.fit(X, y)
        
        # Resize images if needed
        X_processed = self._resize_images(X)
        
        # Normalize pixel values
        if self.normalize:
            X_processed = X_processed / 255.0
        
        # Apply augmentation if requested and available
        if augment and self.augmentation_generator is not None:
            # Note: Augmentation is typically applied during training via generator
            # This is a simple implementation
            pass
        
        # One-hot encode labels if provided
        y_processed = None
        if y is not None:
            y_processed = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        return X_processed, y_processed
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augment: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor and transform data
        
        Args:
            X: Image array
            y: Label array (optional)
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        if y is not None:
            self.fit(X, y)
        return self.transform(X, y, augment=augment)
    
    def _resize_images(self, X: np.ndarray) -> np.ndarray:
        """
        Resize images to target size
        
        Args:
            X: Image array (N, H, W, C)
            
        Returns:
            Resized image array
        """
        if X.shape[1:3] == self.target_size:
            return X
        
        # Use TensorFlow for efficient resizing
        resized_images = []
        for img in X:
            img_tensor = tf.image.resize(img, self.target_size)
            resized_images.append(img_tensor.numpy())
        
        return np.array(resized_images, dtype=np.float32)
    
    def get_augmentation_generator(self) -> Optional[ImageDataGenerator]:
        """
        Get augmentation generator for use during training
        
        Returns:
            ImageDataGenerator or None
        """
        return self.augmentation_generator
    
    def save(self, file_path: str) -> None:
        """
        Save preprocessor to disk
        
        Args:
            file_path: Path to save preprocessor
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor state
        preprocessor_state = {
            'target_size': self.target_size,
            'normalize': self.normalize,
            'augmentation': self.augmentation,
            'num_classes': self.num_classes,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_state, file_path)
        print(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'Preprocessor':
        """
        Load preprocessor from disk
        
        Args:
            file_path: Path to preprocessor file
            
        Returns:
            Loaded Preprocessor instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {file_path}")
        
        preprocessor_state = joblib.load(file_path)
        
        preprocessor = cls(
            target_size=preprocessor_state['target_size'],
            normalize=preprocessor_state['normalize'],
            augmentation=preprocessor_state['augmentation']
        )
        preprocessor.num_classes = preprocessor_state['num_classes']
        preprocessor.is_fitted = preprocessor_state['is_fitted']
        
        print(f"Preprocessor loaded from {file_path}")
        return preprocessor
