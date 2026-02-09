"""
Deep Neural Network model architecture for image classification
Implements a deep CNN with configurable layers and hyperparameters
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Dict, Any


def create_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    config: Dict[str, Any]
) -> keras.Model:
    """
    Create a deep CNN model for image classification
    
    Architecture:
    - Multiple convolutional blocks (Conv2D + BatchNorm + MaxPooling)
    - Dropout regularization
    - Dense layers for classification
    - Minimum 3-5 hidden layers as required
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        config: Configuration dictionary with hyperparameters:
            - learning_rate: Learning rate for optimizer
            - dropout_rate: Dropout rate (0.1-0.5)
            - num_conv_layers: Number of convolutional blocks (default: 4)
            - num_dense_layers: Number of dense layers (default: 2)
            - dense_units: Number of units in dense layers (default: 512)
            - batch_norm: Whether to use batch normalization (default: True)
            - l2_reg: L2 regularization strength (default: 1e-4)
    
    Returns:
        Compiled Keras model
    """
    # Extract hyperparameters from config
    learning_rate = config.get('learning_rate', 0.001)
    dropout_rate = config.get('dropout_rate', 0.3)
    num_conv_layers = config.get('num_conv_layers', 4)
    num_dense_layers = config.get('num_dense_layers', 2)
    dense_units = config.get('dense_units', 512)
    batch_norm = config.get('batch_norm', True)
    l2_reg = config.get('l2_reg', 1e-4)
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Convolutional blocks
    # Each block: Conv2D -> BatchNorm -> Activation -> MaxPooling -> Dropout
    filters = 32  # Start with 32 filters
    
    for i in range(num_conv_layers):
        # Convolutional layer
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'conv2d_{i+1}'
        )(x)
        
        # Batch normalization
        if batch_norm:
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
        
        # Activation
        x = layers.Activation('relu', name=f'activation_{i+1}')(x)
        
        # Max pooling (except for last layer)
        if i < num_conv_layers - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f'maxpool_{i+1}')(x)
        
        # Dropout
        x = layers.Dropout(dropout_rate * 0.5, name=f'dropout_conv_{i+1}')(x)
        
        # Double filters every 2 layers
        if (i + 1) % 2 == 0:
            filters *= 2
    
    # Flatten for dense layers
    x = layers.Flatten(name='flatten')(x)
    
    # Dense layers
    for i in range(num_dense_layers):
        x = layers.Dense(
            units=dense_units,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        )(x)
        
        if batch_norm:
            x = layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
        
        x = layers.Activation('relu', name=f'dense_activation_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        # Reduce units in subsequent layers
        if i < num_dense_layers - 1:
            dense_units = dense_units // 2
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        # Multi-class classification
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'categorical_crossentropy'
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='deep_cnn_classifier')
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get model summary as string
    
    Args:
        model: Keras model
        
    Returns:
        Model summary string
    """
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)


def count_trainable_parameters(model: keras.Model) -> int:
    """
    Count total trainable parameters in model
    
    Args:
        model: Keras model
        
    Returns:
        Number of trainable parameters
    """
    return sum([tf.size(w).numpy() for w in model.trainable_variables])
