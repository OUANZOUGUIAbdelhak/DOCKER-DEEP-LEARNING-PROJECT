"""
Model evaluation script
Evaluates trained model on test set and generates metrics
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import tensorflow as tf
from tensorflow import keras
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.utils.helpers import save_json, load_json


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--model-path', type=str, default='models/best_model.h5',
                        help='Path to trained model')
    parser.add_argument('--preprocessor-path', type=str, default='models/preprocessor.pkl',
                        help='Path to preprocessor')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Path to models directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, save_path: Path):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_training_history(history_path: Path, save_path: Path):
    """
    Plot training curves from history
    
    Args:
        history_path: Path to training history JSON
        save_path: Path to save plot
    """
    history = load_json(str(history_path))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


def generate_prediction_samples(model, X_sample, y_sample, class_names, save_path: Path):
    """
    Generate sample predictions for visualization
    
    Args:
        model: Trained model
        X_sample: Sample images
        y_sample: True labels
        class_names: List of class names
        save_path: Path to save visualization
    """
    predictions = model.predict(X_sample, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # Create visualization
    num_samples = min(len(X_sample), 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = X_sample[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax.imshow(img.astype(np.uint8))
        
        # Add prediction info
        true_label = class_names[y_sample[i]]
        pred_label = class_names[predicted_classes[i]]
        confidence = confidence_scores[i]
        
        color = 'green' if y_sample[i] == predicted_classes[i] else 'red'
        ax.set_title(
            f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
            color=color,
            fontsize=8
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Sample predictions saved to {save_path}")


def evaluate_model(
    model_path: str,
    preprocessor_path: str,
    data_dir: str,
    models_dir: str,
    batch_size: int = 32
):
    """
    Evaluate model on test set
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
        data_dir: Path to data directory
        models_dir: Path to models directory
        batch_size: Batch size for evaluation
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Load preprocessor
    print(f"\nLoading preprocessor from {preprocessor_path}...")
    preprocessor = Preprocessor.load(preprocessor_path)
    print("✓ Preprocessor loaded")
    
    # Load test data
    print("\nLoading test data...")
    loader = DataLoader(data_dir)
    X_test, y_test = loader.load_test_data()
    
    if len(X_test) == 0:
        print("⚠ No test data found. Using validation set for evaluation.")
        X_train, X_val, y_train, y_val = loader.load_train_data()
        X_test = X_val
        y_test = y_val
    
    print(f"Test set size: {len(X_test)} samples")
    
    # Preprocess test data
    print("\nPreprocessing test data...")
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # Get class names
    class_names = loader.get_class_names()
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test_processed, batch_size=batch_size, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_processed, axis=1)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Classification report
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        output_dict=True
    )
    
    # Print results
    print("\n" + "-"*60)
    print("EVALUATION RESULTS")
    print("-"*60)
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print("-"*60)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'classification_report': report
    }
    
    metrics_path = Path(models_dir) / 'test_metrics.json'
    save_json(metrics, str(metrics_path))
    print(f"\n✓ Test metrics saved to {metrics_path}")
    
    # Plot confusion matrix
    cm_path = Path(models_dir) / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Plot training history if available
    history_path = Path(models_dir) / 'training_history.json'
    if history_path.exists():
        curves_path = Path(models_dir) / 'training_curves.png'
        plot_training_history(history_path, curves_path)
    
    # Generate sample predictions
    num_samples = min(10, len(X_test))
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = true_classes[sample_indices]
    
    samples_path = Path(models_dir) / 'sample_predictions.png'
    generate_prediction_samples(model, X_sample, y_sample, class_names, samples_path)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    return metrics


def main():
    """Main evaluation function"""
    args = parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
