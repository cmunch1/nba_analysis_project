import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List
from .base_trainer import BaseTrainer
from .trainer_utils import TrainerUtils
from ml_framework.framework.data_classes import ModelTrainingResults
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

class NeuralNetwork(nn.Module):
    """Simple feedforward neural network for binary classification."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_rate: float = 0.2):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class PyTorchTrainer(BaseTrainer):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """Initialize PyTorch trainer with configuration and dependencies."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.utils = TrainerUtils(app_logger, error_handler)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.app_logger.structured_log(
            logging.INFO, 
            "PyTorchTrainer initialized successfully",
            trainer_type=type(self).__name__,
            device=str(self.device)
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a PyTorch neural network with enhanced tracking."""
        try:
            self.app_logger.structured_log(
                logging.INFO, 
                "Starting PyTorch training", 
                input_shape=X_train.shape,
                device=str(self.device)
            )
            
            # Prepare data
            train_loader, val_loader = self._prepare_data(X_train, y_train, X_val, y_val, model_params)
            
            # Initialize model
            model = self._initialize_model(X_train.shape[1], model_params)
            model.to(self.device)
            
            # Initialize optimizer and loss function
            optimizer = self._get_optimizer(model, model_params)
            criterion = nn.BCELoss()
            
            # Store model configuration
            results.model = model
            results.feature_names = X_train.columns.tolist()
            results.model_params = model_params
            results.categorical_features = self.config.categorical_features if hasattr(self.config, 'categorical_features') else []
            
            # Training loop with learning curve tracking
            train_losses, val_losses = self._training_loop(
                model, train_loader, val_loader, optimizer, criterion, model_params, results
            )
            
            # Generate final predictions
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
                predictions = model(X_val_tensor).cpu().numpy().flatten()
            
            results.predictions = predictions
            results.feature_data = X_val
            results.target_data = y_val
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Generated predictions", 
                predictions_shape=results.predictions.shape,
                predictions_mean=float(np.mean(results.predictions))
            )
            
            # Calculate feature importance using permutation importance
            self._calculate_feature_importance(model, X_val, y_val, results)
            
            # Calculate SHAP values if configured
            if hasattr(self.config, 'calculate_shap_values') and self.config.calculate_shap_values:
                self._calculate_shap_values(model, X_val, y_val, results)
            
            return results

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in PyTorch training",
                original_error=str(e),
                input_shape=X_train.shape
            )

    def _prepare_data(self, X_train, y_train, X_val, y_val, model_params: Dict) -> Tuple[DataLoader, DataLoader]:
        """Prepare PyTorch data loaders."""
        batch_size = model_params.get('batch_size', 64)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    def _initialize_model(self, input_size: int, model_params: Dict) -> NeuralNetwork:
        """Initialize the neural network model."""
        hidden_sizes = model_params.get('hidden_sizes', [128, 64, 32])
        dropout_rate = model_params.get('dropout_rate', 0.2)
        
        return NeuralNetwork(input_size, hidden_sizes, dropout_rate)

    def _get_optimizer(self, model: nn.Module, model_params: Dict) -> optim.Optimizer:
        """Get the optimizer for training."""
        optimizer_name = model_params.get('optimizer', 'adam').lower()
        learning_rate = model_params.get('learning_rate', 0.001)
        weight_decay = model_params.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = model_params.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _training_loop(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      optimizer: optim.Optimizer, criterion: nn.Module, model_params: Dict,
                      results: ModelTrainingResults) -> Tuple[List[float], List[float]]:
        """Main training loop with learning curve tracking."""
        epochs = model_params.get('epochs', 100)
        early_stopping_patience = model_params.get('early_stopping_patience', 10)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.app_logger.structured_log(
                        logging.INFO,
                        "Early stopping triggered",
                        epoch=epoch,
                        best_val_loss=best_val_loss
                    )
                    break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.app_logger.structured_log(
                    logging.INFO,
                    "Training progress",
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss
                )
            
            # Store learning curve data
            if hasattr(self.config, 'generate_learning_curve_data') and self.config.generate_learning_curve_data:
                results.learning_curve_data.add_iteration(
                    train_score=-train_loss,  # Negative because loss (lower is better)
                    val_score=-val_loss,
                    iteration=epoch
                )
        
        results.learning_curve_data.metric_name = 'loss'
        return train_losses, val_losses

    def _calculate_feature_importance(self, model: nn.Module, X_val, y_val, results: ModelTrainingResults) -> None:
        """Calculate feature importance using permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
            
            # Create a wrapper function for the model
            def model_predict(X):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    return model(X_tensor).cpu().numpy().flatten()
            
            # Calculate permutation importance
            r = permutation_importance(
                model_predict,
                X_val.values,
                y_val.values,
                n_repeats=5,
                random_state=self.config.random_state if hasattr(self.config, 'random_state') else 42,
                scoring='neg_log_loss'
            )
            
            results.feature_importance_scores = r.importances_mean
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Calculated feature importance",
                num_features_with_importance=np.sum(r.importances_mean > 0)
            )
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING, 
                "Failed to calculate feature importance",
                error=str(e)
            )
            results.feature_importance_scores = np.zeros(len(results.feature_names))

    def _calculate_shap_values(self, model: nn.Module, X_val, y_val, results: ModelTrainingResults) -> None:
        """Calculate SHAP values using SHAP library."""
        try:
            import shap
            
            # Create a wrapper for SHAP
            def model_predict(X):
                model.eval()
                with torch.no_grad():
                    if isinstance(X, np.ndarray):
                        X_tensor = torch.FloatTensor(X).to(self.device)
                    else:
                        X_tensor = X.to(self.device)
                    return model(X_tensor).cpu().numpy().flatten()
            
            # Use a subset for background if dataset is large
            background_size = min(100, len(X_val))
            background = shap.sample(X_val, background_size)
            
            # Create explainer
            explainer = shap.KernelExplainer(model_predict, background)
            
            # Calculate SHAP values for a subset if dataset is large
            shap_sample_size = min(500, len(X_val))
            sample_indices = np.random.choice(len(X_val), shap_sample_size, replace=False)
            X_shap = X_val.iloc[sample_indices]
            
            shap_values = explainer.shap_values(X_shap.values)
            
            # Store SHAP values (expand to full size with NaN for non-sampled indices)
            full_shap_values = np.full((len(X_val), X_val.shape[1]), np.nan)
            full_shap_values[sample_indices] = shap_values
            
            results.shap_values = full_shap_values
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Calculated SHAP values",
                shap_values_shape=shap_values.shape,
                sample_size=shap_sample_size
            )
            
        except ImportError:
            self.app_logger.structured_log(
                logging.WARNING, 
                "SHAP library not available. SHAP values not calculated."
            )
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR, 
                "Failed to calculate SHAP values",
                error=str(e)
            )

    def _convert_metric_scores(self, train_score: float, val_score: float, metric_name: str) -> Tuple[float, float]:
        """Convert metric scores to a consistent format (higher is better)."""
        return self.utils._convert_metric_scores(train_score, val_score, metric_name)

    def _process_learning_curve_data(self, evals_result: Dict, results: ModelTrainingResults) -> None:
        """Process and store learning curve data."""
        return self.utils._process_learning_curve_data(evals_result, results)