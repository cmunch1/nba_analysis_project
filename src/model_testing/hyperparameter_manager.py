import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import mlflow
import logging
from dataclasses import dataclass, asdict
from .abstract_model_testing import AbstractHyperparameterManager
from ..config.config import AbstractConfig

@dataclass
class HyperparameterSet:
    """Represents a single set of hyperparameters with metadata"""
    name: str
    params: Dict[str, Any]
    performance_metrics: Dict[str, float]
    creation_date: str
    experiment_id: str
    run_id: str
    model_version: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HyperparameterSet':
        return cls(**data)

class HyperparameterManager(AbstractHyperparameterManager):
    def __init__(self, config: AbstractConfig):
        """
        Initialize the hyperparameter manager.
        
        Args:
            config_path: Path to the hyperparameters JSON file
            storage_dir: Directory to store hyperparameter history
        """
        self.config_path = config.current_hyperparameters
        self.storage_dir = config.hyperparameter_history_dir
        self.logger = logging.getLogger(__name__)
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load current hyperparameters
        self.current_params = self._load_json(self.config_path)
        
    def _load_json(self, path: str) -> Dict:
        """Load JSON configuration file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'model_hyperparameters': {}}
            
    def _save_json(self, data: Dict, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_current_params(self, model_name: str) -> Dict[str, Any]:
        """Get current best hyperparameters for a model."""
        model_params = self.current_params.get('model_hyperparameters', {}).get(model_name, [])
        current_best = next((config['params'] for config in model_params 
                           if config['name'] == 'current_best'), None)
        if current_best is None:
            raise ValueError(f"No current_best parameters found for {model_name}")
        return current_best
        
    def update_best_params(self, 
                          model_name: str, 
                          new_params: Dict[str, Any],
                          metrics: Dict[str, float],
                          experiment_id: str,
                          run_id: str,
                          description: Optional[str] = None) -> None:
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
        # Create new hyperparameter set
        new_set = HyperparameterSet(
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            params=new_params,
            performance_metrics=metrics,
            creation_date=datetime.now().isoformat(),
            experiment_id=experiment_id,
            run_id=run_id,
            description=description
        )
        
        # Save to history
        self._save_to_history(model_name, new_set)
        
        # Update current best if necessary
        if self._is_better_than_current(model_name, metrics):
            self._update_current_best(model_name, new_set)
            
    def _is_better_than_current(self, model_name: str, new_metrics: Dict[str, float]) -> bool:
        """
        Compare new metrics with current best metrics.
        Currently uses AUC as the primary metric for comparison.
        """
        try:
            history = self._load_history(model_name)
            if not history:
                return True
                
            current_best = max(history, key=lambda x: x.performance_metrics.get('auc', 0))
            return new_metrics.get('auc', 0) > current_best.performance_metrics.get('auc', 0)
        except Exception:
            return True
            
    def _update_current_best(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Update the current best parameters in the JSON file."""
        if 'model_hyperparameters' not in self.current_params:
            self.current_params['model_hyperparameters'] = {}
            
        if model_name not in self.current_params['model_hyperparameters']:
            self.current_params['model_hyperparameters'][model_name] = []
            
        # Update or add current_best
        model_params = self.current_params['model_hyperparameters'][model_name]
        current_best_idx = next((i for i, config in enumerate(model_params) 
                               if config['name'] == 'current_best'), None)
                               
        new_config = {
            'name': 'current_best',
            'params': param_set.params,
            'metrics': param_set.performance_metrics,
            'updated_at': param_set.creation_date,
            'experiment_id': param_set.experiment_id,
            'run_id': param_set.run_id
        }
        
        if current_best_idx is not None:
            model_params[current_best_idx] = new_config
        else:
            model_params.append(new_config)
            
        self._save_json(self.current_params, self.config_path)
        
    def _save_to_history(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Save hyperparameter set to history."""
        history_file = os.path.join(self.storage_dir, f"{model_name}_history.json")
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
            
        history.append(asdict(param_set))
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    def _load_history(self, model_name: str) -> List[HyperparameterSet]:
        """Load hyperparameter history for a model."""
        history_file = os.path.join(self.storage_dir, f"{model_name}_history.json")
        
        try:
            with open(history_file, 'r') as f:
                history_data = json.load(f)
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
                     
    def export_to_mlflow(self, model_name: str) -> None:
        """Export hyperparameter history to MLflow."""
        history = self._load_history(model_name)
        
        for param_set in history:
            with mlflow.start_run(run_id=param_set.run_id):
                mlflow.log_params(param_set.params)
                mlflow.log_metrics(param_set.performance_metrics)
                mlflow.set_tags({
                    'model_name': model_name,
                    'parameter_set_name': param_set.name,
                    'creation_date': param_set.creation_date
                })