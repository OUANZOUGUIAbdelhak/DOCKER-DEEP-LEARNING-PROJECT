"""
Particle Swarm Optimization (PSO) for hyperparameter tuning
Implements PSO algorithm to find optimal hyperparameters
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.dnn_model import create_model
from src.utils.helpers import set_seed, save_json, load_json


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for hyperparameter tuning
    
    Optimizes:
    - Learning rate
    - Batch size
    - Dropout rate
    - Dense units
    - Number of layers
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        n_particles: int = 10,
        n_iterations: int = 20,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5
    ):
        """
        Initialize PSO optimizer
        
        Args:
            bounds: Dictionary of hyperparameter bounds
                Example: {
                    'learning_rate': (1e-5, 1e-2),
                    'batch_size': (16, 128),
                    'dropout_rate': (0.1, 0.5),
                    'dense_units': (64, 512)
                }
            n_particles: Number of particles in swarm
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive parameter (personal best)
            c2: Social parameter (global best)
        """
        self.bounds = bounds
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Hyperparameter names
        self.param_names = list(bounds.keys())
        self.n_params = len(self.param_names)
        
        # Initialize particles
        self.particles = np.random.uniform(
            low=[bounds[name][0] for name in self.param_names],
            high=[bounds[name][1] for name in self.param_names],
            size=(n_particles, self.n_params)
        )
        
        # Initialize velocities
        self.velocities = np.random.uniform(
            low=-1,
            high=1,
            size=(n_particles, self.n_params)
        )
        
        # Personal bests
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.full(n_particles, -np.inf)
        
        # Global best
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        # History
        self.history = []
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, Any]:
        """
        Convert parameter array to dictionary
        
        Args:
            params: Parameter array
            
        Returns:
            Parameter dictionary
        """
        param_dict = {}
        for i, name in enumerate(self.param_names):
            value = params[i]
            
            # Round batch_size and dense_units to integers
            if name in ['batch_size', 'dense_units', 'num_conv_layers', 'num_dense_layers']:
                value = int(np.round(value))
            else:
                value = float(value)
            
            param_dict[name] = value
        
        return param_dict
    
    def fitness_function(
        self,
        params: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20
    ) -> float:
        """
        Evaluate fitness of a set of hyperparameters
        
        Args:
            params: Hyperparameter array
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs for quick training
            
        Returns:
            Validation accuracy (fitness score)
        """
        # Convert params to dict
        param_dict = self._params_to_dict(params)
        
        # Ensure batch_size is valid
        batch_size = max(16, min(128, param_dict.get('batch_size', 32)))
        
        try:
            # Create model
            input_shape = X_train.shape[1:]
            num_classes = y_train.shape[1]
            
            # Build config for model
            config = {
                'learning_rate': param_dict.get('learning_rate', 0.001),
                'dropout_rate': param_dict.get('dropout_rate', 0.3),
                'dense_units': param_dict.get('dense_units', 512),
                'num_conv_layers': param_dict.get('num_conv_layers', 4),
                'num_dense_layers': param_dict.get('num_dense_layers', 2),
                'batch_norm': True,
                'l2_reg': 1e-4
            }
            
            model = create_model(input_shape, num_classes, config)
            
            # Train for limited epochs
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Get best validation accuracy
            best_val_accuracy = max(history.history['val_accuracy'])
            
            # Clean up
            del model
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
            return float(best_val_accuracy)
        
        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return -1.0  # Return low fitness for invalid configurations
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs_per_eval: int = 20
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run PSO optimization
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs_per_eval: Number of epochs per fitness evaluation
            
        Returns:
            Tuple of (best_params, best_score)
        """
        print("\n" + "="*60)
        print("PARTICLE SWARM OPTIMIZATION")
        print("="*60)
        print(f"Particles: {self.n_particles}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Hyperparameters: {self.param_names}")
        print("="*60)
        
        # Initial evaluation
        print("\nInitial evaluation...")
        for i in range(self.n_particles):
            score = self.fitness_function(
                self.particles[i],
                X_train, y_train, X_val, y_val,
                epochs=epochs_per_eval
            )
            
            self.personal_best_scores[i] = score
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()
            
            print(f"Particle {i+1}/{self.n_particles}: Score = {score:.4f}")
        
        print(f"\nInitial best score: {self.global_best_score:.4f}")
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration+1}/{self.n_iterations}")
            print(f"{'='*60}")
            
            iteration_best_score = -np.inf
            
            for i in range(self.n_particles):
                # Evaluate fitness
                score = self.fitness_function(
                    self.particles[i],
                    X_train, y_train, X_val, y_val,
                    epochs=epochs_per_eval
                )
                
                # Update personal best
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                
                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()
                
                if score > iteration_best_score:
                    iteration_best_score = score
                
                # Update velocity
                r1 = np.random.random(self.n_params)
                r2 = np.random.random(self.n_params)
                
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social = self.c2 * r2 * (self.global_best_position - self.particles[i])
                
                self.velocities[i] = (
                    self.w * self.velocities[i] + cognitive + social
                )
                
                # Update position
                self.particles[i] += self.velocities[i]
                
                # Apply bounds
                for j, name in enumerate(self.param_names):
                    low, high = self.bounds[name]
                    self.particles[i][j] = np.clip(self.particles[i][j], low, high)
            
            # Record history
            self.history.append({
                'iteration': iteration + 1,
                'best_score': float(self.global_best_score),
                'iteration_best': float(iteration_best_score)
            })
            
            print(f"Best score so far: {self.global_best_score:.4f}")
            print(f"Iteration best: {iteration_best_score:.4f}")
        
        # Convert best position to dict
        best_params = self._params_to_dict(self.global_best_position)
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"Best score: {self.global_best_score:.4f}")
        print(f"Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        return best_params, self.global_best_score


def main():
    """Main optimization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PSO Hyperparameter Optimization')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--n-particles', type=int, default=10, help='Number of particles')
    parser.add_argument('--n-iterations', type=int, default=20, help='Number of iterations')
    parser.add_argument('--epochs-per-eval', type=int, default=20, help='Epochs per evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    print("Loading data...")
    loader = DataLoader(args.data_dir)
    X_train, X_val, y_train, y_val = loader.load_train_data(validation_split=0.15)
    
    # Preprocess
    print("Preprocessing data...")
    preprocessor = Preprocessor(target_size=(224, 224), normalize=True, augmentation=False)
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
    
    # Use subset for faster optimization (optional)
    # X_train_processed = X_train_processed[:1000]
    # y_train_processed = y_train_processed[:1000]
    
    # Define bounds
    bounds = {
        'learning_rate': (1e-5, 1e-2),
        'batch_size': (16, 128),
        'dropout_rate': (0.1, 0.5),
        'dense_units': (64, 512),
        'num_conv_layers': (3, 5),
        'num_dense_layers': (1, 3)
    }
    
    # Create optimizer
    optimizer = ParticleSwarmOptimizer(
        bounds=bounds,
        n_particles=args.n_particles,
        n_iterations=args.n_iterations
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(
        X_train_processed,
        y_train_processed,
        X_val_processed,
        y_val_processed,
        epochs_per_eval=args.epochs_per_eval
    )
    
    # Load baseline metrics if available
    baseline_accuracy = None
    metrics_path = Path(args.models_dir) / 'metrics.json'
    if metrics_path.exists():
        baseline_metrics = load_json(str(metrics_path))
        baseline_accuracy = baseline_metrics.get('best_val_accuracy', None)
    
    # Save results
    results = {
        'baseline_accuracy': float(baseline_accuracy) if baseline_accuracy else None,
        'optimized_accuracy': float(best_score),
        'improvement': float(best_score - baseline_accuracy) if baseline_accuracy else None,
        'best_params': best_params,
        'optimization_history': optimizer.history,
        'n_particles': args.n_particles,
        'n_iterations': args.n_iterations
    }
    
    results_path = Path(args.models_dir) / 'optimization_results.json'
    save_json(results, str(results_path))
    print(f"\n✓ Optimization results saved to {results_path}")


if __name__ == '__main__':
    main()
