"""
Hyperparameter manager with support for nested configurations and proper dependency injection.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from .base_hyperparams_manager import BaseHyperparamsManager

@dataclass
class HyperparameterSet:
    """Represents a single set of hyperparameters with metadata"""
    name: str
    params: Dict[str, Any]
    num_boost_round: int
    performance_metrics: Dict[str, float]
    creation_date: str
    experiment_id: str
    run_id: str
    early_stopping: int
    enable_categorical: bool
    categorical_features: List[str]
    model_version: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HyperparameterSet':
        return cls(**data)

class HyperparamsManager(BaseHyperparamsManager):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, 
                 app_file_handler: BaseAppFileHandler, error_handler: BaseErrorHandler):
        """
        Initialize the hyperparameter manager with injected dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            app_file_handler: File handling utility
            error_handler: Error handling utility
        """
        self.config = config
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler
        self.error_handler = error_handler
        
        # Get paths from config
        self.config_path = self._get_config_path()
        self.storage_dir = self._get_storage_dir()
        
        # Ensure storage directory exists
        self.app_file_handler.ensure_directory(self.storage_dir)
        
        # Load current parameters
        self.current_params = self.app_file_handler.read_json(self.config_path)
        
        self.app_logger.structured_log(logging.INFO, "HyperParamsManager initialized",
                                     config_path=self.config_path,
                                     storage_dir=self.storage_dir)

    def _get_config_path(self) -> str:
        """Get hyperparameter config path from nested config."""
        try:
            return self.config.hyperparameters.config_path
        except AttributeError:
            return self.config.current_hyperparameters  # Fallback to old config structure

    def _get_storage_dir(self) -> str:
        """Get hyperparameter storage directory from nested config."""
        try:
            return self.config.hyperparameters.storage_dir
        except AttributeError:
            return self.config.hyperparameter_history_dir  # Fallback to old config structure

    def get_current_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get the current hyperparameters for a model, handling nested configurations.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict containing model parameters
        """
        self.app_logger.structured_log(logging.INFO, "Getting current parameters",
                                     model_name=model_name)
        
        try:
            # Get model-specific config
            model_config = self._get_model_config(model_name)
            
            # Determine which parameter set to use
            use_baseline = (not model_config.perform_hyperparameter_optimization and 
                          model_config.use_baseline_hyperparameters)
            param_name = 'baseline' if use_baseline else 'current_best'
            
            # Load and validate parameters
            params = self._load_parameter_set(model_name, param_name)
            if params is None:
                raise ValueError(f"No {param_name} parameters found for {model_name}")
                
            return params
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'hyperparameter_management',
                f"Error getting current parameters for {model_name}",
                original_error=str(e)
            )

    def _get_model_config(self, model_name: str) -> Any:
        """
        Get model-specific configuration, handling nested paths.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model-specific configuration object
        """
        try:
            # Handle SKLearn models
            if model_name.startswith('SKLearn_'):
                base_model = model_name.split('_')[1]
                return getattr(self.config.models.SKLearn, base_model)
            # Handle other models
            return getattr(self.config.models, model_name)
        except AttributeError:
            raise ValueError(f"No configuration found for model: {model_name}")

    def _load_parameter_set(self, model_name: str, param_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific parameter set for a model.
        
        Args:
            model_name: Name of the model
            param_name: Name of the parameter set to load
            
        Returns:
            Dictionary containing parameters if found, None otherwise
        """
        model_params = self.current_params.get('model_hyperparameters', {}).get(model_name, [])
        return next((config['params'] for config in model_params 
                    if config['name'] == param_name), None)

    def update_best_params(self, model_name: str, new_params: Dict[str, Any],
                          metrics: Dict[str, float], experiment_id: str,
                          run_id: Optional[str] = None, description: Optional[str] = None) -> None:
        """
        Update the best hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            new_params: New hyperparameters to evaluate
            metrics: Performance metrics for the new parameters
            experiment_id: MLflow experiment ID
            run_id: Optional MLflow run ID
            description: Optional description of the parameter set
        """
        self.app_logger.structured_log(logging.INFO, "Updating best parameters",
                                     model_name=model_name,
                                     metrics=metrics,
                                     run_id=run_id)
        
        try:
            # Generate run_id if none provided
            if run_id is None:
                run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get model-specific config
            model_config = self._get_model_config(model_name)
            
            # Create new hyperparameter set
            new_set = self._create_parameter_set(
                model_name=model_name,
                model_config=model_config,
                params=new_params,
                metrics=metrics,
                experiment_id=experiment_id,
                run_id=run_id,
                description=description
            )
            
            # Save to history
            self._save_to_history(model_name, new_set)
            
            # Update current best if necessary
            self._update_current_best(model_name, new_set)
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'hyperparameter_management',
                f"Error updating best parameters for {model_name}",
                original_error=str(e)
            )

    def _create_parameter_set(self, model_name: str, model_config: Any,
                            params: Dict[str, Any], metrics: Dict[str, float],
                            experiment_id: str, run_id: str,
                            description: Optional[str] = None) -> HyperparameterSet:
        """Create a new hyperparameter set with model-specific settings."""
        return HyperparameterSet(
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            params=params,
            performance_metrics=metrics,
            creation_date=datetime.now().isoformat(),
            experiment_id=experiment_id,
            run_id=run_id,
            description=description,
            num_boost_round=getattr(model_config, 'num_boost_round', 0),
            early_stopping=getattr(model_config, 'early_stopping_rounds', 0),
            enable_categorical=getattr(model_config, 'enable_categorical', False),
            categorical_features=self.config.categorical_features
        )

    def _save_to_history(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Save hyperparameter set to history file."""
        history_file = self.app_file_handler.join_paths(
            self.storage_dir, 
            f"{model_name}_history.json"
        )
        
        try:
            history = self.app_file_handler.read_json(history_file)
        except FileNotFoundError:
            history = []
            
        history.append(asdict(param_set))
        self.app_file_handler.write_json(history, history_file)

    def _update_current_best(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Update current best parameters based on evaluation metric."""
        history = self._load_history(model_name)
        
        # Get evaluation metric
        eval_metric = self._get_eval_metric(param_set.params)
        
        # Find best performing parameter set
        best_set = max(history, 
                      key=lambda x: x.performance_metrics.get(eval_metric, 0))
        
        # Update current parameters
        if 'model_hyperparameters' not in self.current_params:
            self.current_params['model_hyperparameters'] = {}
            
        if model_name not in self.current_params['model_hyperparameters']:
            self.current_params['model_hyperparameters'][model_name] = []
            
        new_config = {
            'name': 'current_best',
            'params': best_set.params,
            'metrics': best_set.performance_metrics,
            'updated_at': best_set.creation_date,
            'experiment_id': best_set.experiment_id,
            'run_id': best_set.run_id,
            'num_boost_round': best_set.num_boost_round,
            'early_stopping': best_set.early_stopping,
            'enable_categorical': best_set.enable_categorical,
            'categorical_features': best_set.categorical_features
        }
        
        # Update or append new configuration
        model_params = self.current_params['model_hyperparameters'][model_name]
        current_best_idx = next((i for i, config in enumerate(model_params) 
                               if config['name'] == 'current_best'), None)
        
        if current_best_idx is not None:
            model_params[current_best_idx] = new_config
        else:
            model_params.append(new_config)
            
        self.app_file_handler.write_json(self.current_params, self.config_path)

    def _load_history(self, model_name: str) -> List[HyperparameterSet]:
        """Load hyperparameter history for a model."""
        history_file = self.app_file_handler.join_paths(
            self.storage_dir,
            f"{model_name}_history.json"
        )
        
        try:
            history_data = self.app_file_handler.read_json(history_file)
            return [HyperparameterSet.from_dict(item) for item in history_data]
        except FileNotFoundError:
            return []

    def _get_eval_metric(self, params: Dict[str, Any]) -> str:
        """Extract evaluation metric from parameters."""
        # Check different framework metric parameter names
        eval_metric = (
            params.get('eval_metric') or      # XGBoost, CatBoost
            params.get('metric') or           # LightGBM
            params.get('scoring') or          # Sklearn
            'auc'                            # Default fallback
        )
        
        # Handle different metric formats
        if isinstance(eval_metric, (list, tuple)):
            eval_metric = eval_metric[0]
        elif isinstance(eval_metric, dict):
            eval_metric = next(iter(eval_metric.values()))
            
        return eval_metric
