"""
Hyperparameter manager with support for nested configurations and proper dependency injection.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

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
        
        # Initialize current_params with empty structure
        self.current_params = {"model_hyperparameters": {}}
        
        # Create storage directory if needed
        self.storage_dir = self._get_storage_dir()
        self.app_file_handler.ensure_directory(self.storage_dir)
        
        # Log initialization
        self.app_logger.structured_log(logging.INFO, "HyperParamsManager initialized",
                                    storage_dir=self.storage_dir)

    def _get_storage_dir(self) -> str:
        """Get hyperparameter storage directory from nested config."""
        try:
            if hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, 'storage_dir'):
                return self.config.hyperparameters.storage_dir
            elif hasattr(self.config, 'hyperparameter_history_dir'):
                return self.config.hyperparameter_history_dir
            else:
                # Default fallback
                return "hyperparameter_history"
        except AttributeError as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Could not find hyperparameter storage directory in config, using default",
                error=str(e)
            )
            return "hyperparameter_history"

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
            use_baseline = False
            if model_config and hasattr(model_config, 'perform_hyperparameter_optimization'):
                use_baseline = (not model_config.perform_hyperparameter_optimization and 
                              hasattr(model_config, 'use_baseline_hyperparameters') and
                              model_config.use_baseline_hyperparameters)
            
            # Load parameters from model-specific JSON file
            params = self._load_model_params(model_name, "baseline" if use_baseline else "current_best")
            
            if params is None:
                self.app_logger.structured_log(
                    logging.WARNING,
                    f"No parameters found for {model_name}, using default from config"
                )
                # Return empty default params
                return {}
                
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
                base_model = model_name.split('_')[1].lower()
                return getattr(self.config.models.SKLearn, base_model)
            # Handle other models
            return getattr(self.config.models, model_name.lower())
        except AttributeError:
            self.app_logger.structured_log(
                logging.WARNING,
                f"No configuration found for model: {model_name}"
            )
            # Return None instead of raising an error to allow fallback
            return None

    def _get_model_hyperparams_dir(self, model_name: str) -> Path:
        """
        Get the directory path for a model's hyperparameters based on config structure.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the model's hyperparameter directory
        """
        # Get hyperparams base directory from config
        if hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, 'config_dir'):
            base_dir = self.config.hyperparameters.config_dir
        else:
            # Default to the configs/hyperparameters structure
            base_dir = Path("configs") / "hyperparameters"
            
        # Handle SKLearn models which have a nested structure
        if model_name.startswith('SKLearn_'):
            base_model = model_name.split('_')[1].lower()
            return base_dir / "sklearn" / base_model
        else:
            return base_dir / model_name.lower()

    def _load_model_params(self, model_name: str, param_type: str = "current_best") -> Optional[Dict[str, Any]]:
        """
        Load parameters from model-specific JSON files in the config structure.
        
        Args:
            model_name: Name of the model
            param_type: Type of parameters to load ("current_best" or "baseline")
            
        Returns:
            Dictionary of parameters or None if not found
        """
        try:
            # Get the model's hyperparameter directory
            model_dir = self._get_model_hyperparams_dir(model_name)
            
            # First try the requested parameter type
            param_file = model_dir / f"{param_type}.json"
            
            # Check if file exists and try to load it
            if self.app_file_handler.join_paths(param_file).exists():
                self.app_logger.structured_log(
                    logging.INFO,
                    f"Loading {param_type} parameters for {model_name}",
                    file_path=str(param_file)
                )
                try:
                    params_data = self.app_file_handler.read_json(param_file)
                    return params_data.get("params", {})
                except Exception as e:
                    self.app_logger.structured_log(
                        logging.WARNING,
                        f"Error reading {param_file}",
                        error=str(e)
                    )
            
            # If requested type fails or doesn't exist, try the alternate
            alternate_type = "baseline" if param_type == "current_best" else "current_best"
            alternate_file = model_dir / f"{alternate_type}.json"
            
            if self.app_file_handler.join_paths(alternate_file).exists():
                self.app_logger.structured_log(
                    logging.INFO,
                    f"Loading {alternate_type} parameters for {model_name} as fallback",
                    file_path=str(alternate_file)
                )
                try:
                    params_data = self.app_file_handler.read_json(alternate_file)
                    return params_data.get("params", {})
                except Exception as e:
                    self.app_logger.structured_log(
                        logging.WARNING,
                        f"Error reading fallback file {alternate_file}",
                        error=str(e)
                    )
            
            # If we get here, neither file was found or loadable
            self.app_logger.structured_log(
                logging.WARNING,
                f"No parameter files found for {model_name}",
                attempted_paths=[str(param_file), str(alternate_file)]
            )
            return None
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                f"Error loading model parameters for {model_name}",
                error=str(e)
            )
            return None

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
            
            # Update current best
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
        # Get default values for params that might not be in model_config
        num_boost_round = 100
        early_stopping = 10
        enable_categorical = False
        categorical_features = []
        
        # Extract values from model_config if it exists
        if model_config is not None:
            if hasattr(model_config, 'num_boost_round'):
                num_boost_round = model_config.num_boost_round
            if hasattr(model_config, 'early_stopping'):
                early_stopping = model_config.early_stopping
            if hasattr(model_config, 'enable_categorical'):
                enable_categorical = model_config.enable_categorical
        
        # Get categorical features from main config
        if hasattr(self.config, 'categorical_features'):
            categorical_features = self.config.categorical_features
        
        return HyperparameterSet(
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            params=params,
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

    def _save_to_history(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Save hyperparameter set to history file."""
        history_file = self.app_file_handler.join_paths(self.storage_dir, f"{model_name}_history.json")
        
        try:
            # Ensure directory exists
            self.app_file_handler.ensure_directory(self.storage_dir)
            
            # Load existing history or create empty list
            try:
                history = self.app_file_handler.read_json(history_file)
            except FileNotFoundError:
                history = []
            except Exception as e:
                self.app_logger.structured_log(
                    logging.WARNING,
                    f"Error reading history file, creating new one: {str(e)}"
                )
                history = []
                
            # Add new parameter set and save
            history.append(asdict(param_set))
            self.app_file_handler.write_json(history, history_file)
            
            self.app_logger.structured_log(
                logging.INFO,
                f"Saved parameter set to history file",
                history_file=str(history_file)
            )
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                f"Error saving to history file",
                error=str(e),
                history_file=str(history_file)
            )

    def _update_current_best(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Update current best parameters in the appropriate config file."""
        try:
            # Get the model's hyperparameter directory
            model_dir = self._get_model_hyperparams_dir(model_name)
            
            # Ensure directory exists
            self.app_file_handler.ensure_directory(model_dir)
            
            # Create current best param data
            current_best = {
                "name": "current_best",
                "params": param_set.params,
                "metrics": param_set.performance_metrics,
                "updated_at": param_set.creation_date,
                "experiment_id": param_set.experiment_id,
                "run_id": param_set.run_id,
                "num_boost_round": param_set.num_boost_round,
                "early_stopping": param_set.early_stopping,
                "enable_categorical": param_set.enable_categorical,
                "categorical_features": param_set.categorical_features
            }
            
            # Save to current_best.json
            current_best_path = model_dir / "current_best.json"
            self.app_file_handler.write_json(current_best, current_best_path)
            
            self.app_logger.structured_log(
                logging.INFO,
                f"Updated current best parameters",
                file_path=str(current_best_path)
            )
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                f"Error updating current best parameters",
                error=str(e),
                model_name=model_name
            )

    def _load_history(self, model_name: str) -> List[HyperparameterSet]:
        """Load hyperparameter history for a model."""
        history_file = self.app_file_handler.join_paths(self.storage_dir, f"{model_name}_history.json")
        
        try:
            try:
                history_data = self.app_file_handler.read_json(history_file)
                return [HyperparameterSet.from_dict(item) for item in history_data]
            except FileNotFoundError:
                self.app_logger.structured_log(
                    logging.INFO,
                    f"No history file found for {model_name}",
                    history_file=str(history_file)
                )
                return []
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                f"Error loading history for {model_name}",
                error=str(e),
                history_file=str(history_file)
            )
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
