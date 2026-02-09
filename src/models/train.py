"""
Training script for Deep Neural Network
Main entry point for model training
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.dnn_model import create_model, count_trainable_parameters
from src.utils.helpers import set_seed, save_json
from src.utils.config import load_config


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Deep Neural Network')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Path to models directory')
    parser.add_argument('--logs-dir', type=str, default='logs',
                        help='Path to logs directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    return parser.parse_args()


def setup_gpu():
    """Setup and check GPU availability"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
            return False
    else:
        print("⚠ No GPU detected, using CPU")
        return False


def create_callbacks(models_dir: Path, logs_dir: Path) -> list:
    """
    Create training callbacks
    
    Args:
        models_dir: Directory to save models
        logs_dir: Directory to save logs
        
    Returns:
        List of callbacks
    """
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=str(models_dir / 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard for visualization
        TensorBoard(
            log_dir=str(logs_dir / 'tensorboard'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks


def save_training_history(history: keras.callbacks.History, file_path: Path):
    """
    Save training history to JSON
    
    Args:
        history: Training history object
        file_path: Path to save history
    """
    history_dict = {}
    for key in history.history.keys():
        history_dict[key] = [float(val) for val in history.history[key]]
    
    save_json(history_dict, str(file_path))
    print(f"Training history saved to {file_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    logs_dir = Path(args.logs_dir)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup GPU
    use_gpu = setup_gpu() if args.gpu else False
    
    print("\n" + "="*60)
    print("DEEP NEURAL NETWORK TRAINING")
    print("="*60)
    
    # Load configuration if provided
    config = {}
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
    else:
        # Use command-line arguments as config
        config = {
            'learning_rate': args.learning_rate,
            'dropout_rate': args.dropout_rate,
            'num_conv_layers': 4,
            'num_dense_layers': 2,
            'dense_units': 512,
            'batch_norm': True,
            'l2_reg': 1e-4
        }
    
    # Override config with command-line arguments
    config['learning_rate'] = args.learning_rate
    config['dropout_rate'] = args.dropout_rate
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\n" + "-"*60)
    print("Loading data...")
    print("-"*60)
    
    loader = DataLoader(str(data_dir))
    X_train, X_val, y_train, y_val = loader.load_train_data(validation_split=0.15)
    
    # Validate data
    loader.validate_data(X_train, y_train)
    
    # Get class names
    class_names = loader.get_class_names()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Preprocess data
    print("\n" + "-"*60)
    print("Preprocessing data...")
    print("-"*60)
    
    preprocessor = Preprocessor(
        target_size=(224, 224),
        normalize=True,
        augmentation=True
    )
    
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
    
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_val_processed.shape}")
    
    # Save preprocessor
    preprocessor_path = models_dir / 'preprocessor.pkl'
    preprocessor.save(str(preprocessor_path))
    print(f"✓ Preprocessor saved to {preprocessor_path}")
    
    # Create model
    print("\n" + "-"*60)
    print("Creating model...")
    print("-"*60)
    
    input_shape = X_train_processed.shape[1:]
    model = create_model(input_shape, num_classes, config)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    num_params = count_trainable_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    # Create callbacks
    callbacks = create_callbacks(models_dir, logs_dir)
    
    # Train model
    print("\n" + "-"*60)
    print("Training model...")
    print("-"*60)
    
    start_time = time.time()
    
    history = model.fit(
        X_train_processed,
        y_train_processed,
        validation_data=(X_val_processed, y_val_processed),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
    
    # Save training history
    history_path = models_dir / 'training_history.json'
    save_training_history(history, history_path)
    
    # Evaluate on validation set
    print("\n" + "-"*60)
    print("Evaluating model...")
    print("-"*60)
    
    val_metrics = model.evaluate(X_val_processed, y_val_processed, verbose=1)
    val_loss, val_accuracy = val_metrics[0], val_metrics[1]
    
    # Get best validation accuracy from history
    best_val_accuracy = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    
    print(f"\nValidation Results:")
    print(f"  Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Final Validation Loss: {val_loss:.4f}")
    
    # Save metrics
    metrics = {
        'best_val_accuracy': float(best_val_accuracy),
        'best_val_loss': float(best_val_loss),
        'final_val_accuracy': float(val_accuracy),
        'final_val_loss': float(val_loss),
        'training_time_minutes': float(training_time / 60),
        'num_epochs_trained': len(history.history['loss']),
        'num_trainable_params': int(num_params),
        'used_gpu': use_gpu,
        'config': config
    }
    
    metrics_path = models_dir / 'metrics.json'
    save_json(metrics, str(metrics_path))
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Save final model
    final_model_path = models_dir / 'final_model.h5'
    model.save(str(final_model_path))
    print(f"✓ Final model saved to {final_model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: {models_dir / 'best_model.h5'}")
    print(f"Training history: {history_path}")
    print(f"Metrics: {metrics_path}")
    print("="*60)


if __name__ == '__main__':
    main()
