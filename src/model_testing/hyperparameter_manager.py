import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import mlflow
import logging
from dataclasses import dataclass, asdict
from src.logging.logging_utils import structured_log
from .abstract_model_testing import AbstractHyperparameterManager
from ..config.config import AbstractConfig
from src.common.app_file_handling.base_app_file_handler import BaseAppFileHandler

logger = logging.getLogger(__name__)

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

class HyperparameterManager(AbstractHyperparameterManager):
    def __init__(self, config: AbstractConfig, app_file_handler: BaseAppFileHandler):
        """
        Initialize the hyperparameter manager.
        
        Args:
            config_path: Path to the hyperparameters JSON file
            storage_dir: Directory to store hyperparameter history
        """
        self.config_path = config.current_hyperparameters
        self.storage_dir = config.hyperparameter_history_dir
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.app_file_handler = app_file_handler
        
        self.app_file_handler.ensure_directory(self.storage_dir)
        self.current_params = self.app_file_handler.read_json(self.config_path)
        
    def _load_json(self, path: str) -> Dict:
        """Load JSON configuration file."""
        structured_log(self.logger, logging.INFO, "Loading Hyperparameter JSON configuration file",
                      config_path=path)
        try:
            return self.app_file_handler.read_json(path)
        except FileNotFoundError:
            return {'model_hyperparameters': {}}
            
    def _save_json(self, data: Dict, path: str) -> None:
        """Save configuration to JSON file."""
        structured_log(self.logger, logging.INFO, "Saving JSON configuration file",
                    config_path=path,
                    data=data)  
        self.app_file_handler.write_json(data, path)
        structured_log(self.logger, logging.INFO, "Successfully saved JSON configuration")
            
    def get_current_params(self, model_name: str, param_name: str = 'current_best') -> Dict[str, Any]:
        """Get current best hyperparameters for a model."""
        structured_log(self.logger, logging.INFO, "Getting current best hyperparameters for a model",
                      model_name=model_name)
        
        if self.config.use_baseline_hyperparameters and self.config.perform_hyperparameter_optimization == False:
            param_name = 'baseline'
        
        try:
            self.current_params = self.app_file_handler.read_json(self.config_path)
            model_params = self.current_params.get('model_hyperparameters', {}).get(model_name, [])
            current_params = next((config['params'] for config in model_params 
                               if config['name'] == param_name), None)
            if current_params is None:
                raise ValueError(f"No {param_name} parameters found for {model_name}")
            return current_params
        except Exception as e:
            structured_log(self.logger, logging.ERROR, f"Error getting {param_name} hyperparameters",
                          error=str(e))
            raise e
        
    def update_best_params(self, model_name: str, new_params: Dict[str, Any],
                          metrics: Dict[str, float], experiment_id: str,
                          run_id: Optional[str] = None, description: Optional[str] = None) -> None:
        """
        Update the best hyperparameters for a model if the new parameters perform better.
        
        Args:
            model_name: Name of the model
            new_params: New hyperparameters to evaluate
            metrics: Performance metrics for the new parameters
            experiment_id: MLflow experiment ID
            run_id: MLflow run ID
            description: Optional description of the parameter set
        """
        structured_log(self.logger, logging.INFO, 
                      "Updating best parameters",
                      model_name=model_name,
                      metrics=metrics,
                      run_id=run_id)
        
        # Generate run_id if none provided
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if model_name == "xgboost":
            num_boost_round = self.config.XGBoost.num_boost_round   
            early_stopping = self.config.XGBoost.early_stopping_rounds
            enable_categorical = self.config.XGBoost.enable_categorical
            categorical_features = self.config.XGBoost.categorical_features

        elif model_name == "lightgbm":
            num_boost_round = self.config.LightGBM.num_boost_round
            early_stopping = self.config.LightGBM.early_stopping
            categorical_features = self.config.LightGBM.categorical_features

        elif model_name == "catboost":
            num_boost_round = self.config.CatBoost.num_boost_round
            early_stopping = self.config.CatBoost.early_stopping_rounds
            categorical_features = self.config.CatBoost.categorical_features

        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        

        # Create new hyperparameter set
        new_set = HyperparameterSet(
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            params=new_params,
            performance_metrics=metrics,
            creation_date=datetime.now().isoformat(),
            experiment_id=experiment_id,
            run_id=run_id,
            description=description,
            num_boost_round=num_boost_round,
            early_stopping=early_stopping,
            enable_categorical=enable_categorical,
            categorical_features=categorical_features
        )
        
        # Save to history with logging
        try:
            self._save_to_history(model_name, new_set)
            structured_log(self.logger, logging.INFO, 
                          "Saved parameters to history",
                          model_name=model_name)
        except Exception as e:
            structured_log(self.logger, logging.ERROR, 
                          "Failed to save parameters to history",
                          model_name=model_name,
                          error=str(e))
            raise 
        
        # Update current best if necessary
        try:    
            self._update_current_best(model_name, new_set)
            structured_log(self.logger, logging.INFO, 
                            "Updated current best parameters",
                            model_name=model_name)
        except Exception as e:
            structured_log(self.logger, logging.ERROR, 
                          "Failed to update current best parameters",
                          model_name=model_name,
                          error=str(e))
            raise
            
    def _get_eval_metric(self, params: Dict[str, Any]) -> str:
        """
        Extract the evaluation metric from model parameters.
        Handles different metric parameter names for various ML frameworks:
        - XGBoost: 'eval_metric'
        - LightGBM: 'metric'
        - CatBoost: 'eval_metric'
        - Sklearn: 'scoring'
        Falls back to 'auc' if no metric is specified.
        """
        # Check different framework metric parameter names
        eval_metric = (
            params.get('eval_metric') or      # XGBoost, CatBoost
            params.get('metric') or           # LightGBM
            params.get('scoring') or          # Sklearn
            'auc'                            # Default fallback
        )
        
        # Handle cases where metric might be in a list or dict
        if isinstance(eval_metric, (list, tuple)):
            eval_metric = eval_metric[0]  # Use first metric if multiple are specified
        elif isinstance(eval_metric, dict):
            eval_metric = next(iter(eval_metric.values()))  # Take first metric from dict
            
        return eval_metric

            
    def _update_current_best(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Update current best by finding best performing params from history."""
        history = self._load_history(model_name)
        
        # Get the evaluation metric from the last entry's performance metrics
        if history:
            # Use the first metric found in performance_metrics from the last history entry
            eval_metric = next(iter(history[-1].performance_metrics.keys()))
        else:
            # Fallback to getting it from params if history is empty
            eval_metric = self._get_eval_metric(param_set.params)
        
        best_set = max(history, key=lambda x: x.performance_metrics.get(eval_metric, 0))
        
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
        
        model_params = self.current_params['model_hyperparameters'][model_name]
        current_best_idx = next((i for i, config in enumerate(model_params) 
                            if config['name'] == 'current_best'), None)
        
        if current_best_idx is not None:
            model_params[current_best_idx] = new_config
        else:
            model_params.append(new_config)
            
        self.file_handler.write_json(self.current_params, self.config_path)
        
    def _save_to_history(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Save hyperparameter set to history."""
        history_file = self.app_file_handler.join_paths(self.storage_dir, f"{model_name}_history.json")
        
        try:
            history = self.app_file_handler.read_json(history_file)
        except FileNotFoundError:
            history = []
            
        history.append(asdict(param_set))
        self.app_file_handler.write_json(history, history_file)
            
    def _load_history(self, model_name: str) -> List[HyperparameterSet]:
        """Load hyperparameter history for a model."""
        history_file = self.app_file_handler.join_paths(self.storage_dir, f"{model_name}_history.json")
        
        try:
            history_data = self.app_file_handler.read_json(history_file)
            return [HyperparameterSet.from_dict(item) for item in history_data]
        except FileNotFoundError:
            return []
            
    def get_parameter_history(self, model_name: str) -> List[HyperparameterSet]:
        """Get the complete history of hyperparameters for a model."""
        return self._load_history(model_name)
        
    def get_best_parameters(self, model_name: str, 
                          metric: str = 'auc',
                          n_best: int = 5) -> List[HyperparameterSet]:
        """Get the n best performing parameter sets for a model."""
        history = self._load_history(model_name)
        return sorted(history, 
                     key=lambda x: x.performance_metrics.get(metric, 0),
                     reverse=True)[:n_best]
                     
