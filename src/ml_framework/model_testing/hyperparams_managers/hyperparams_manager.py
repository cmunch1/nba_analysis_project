"""
Hyperparameter manager with proper separation of concerns.

This manager:
- Reads baseline hyperparameters from the config object (loaded by ConfigManager)
- Manages dynamic hyperparameters in a separate storage directory
- Tracks hyperparameter history for experiments
- Never modifies the source-controlled config files
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
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
            config: Configuration manager (contains baseline hyperparameters)
            app_logger: Application logger
            app_file_handler: File handling utility
            error_handler: Error handling utility
        """
        self.config = config
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler
        self.error_handler = error_handler

        # Get storage directory for dynamic hyperparameters
        self.storage_dir = self._get_storage_dir()
        self.app_file_handler.ensure_directory(self.storage_dir)

        # Create subdirectories for organization
        self.current_best_dir = Path(self.storage_dir) / "current_best"
        self.history_dir = Path(self.storage_dir) / "history"
        self.app_file_handler.ensure_directory(self.current_best_dir)
        self.app_file_handler.ensure_directory(self.history_dir)

        # Log initialization
        self.app_logger.structured_log(
            logging.INFO,
            "HyperParamsManager initialized",
            storage_dir=self.storage_dir,
            current_best_dir=str(self.current_best_dir),
            history_dir=str(self.history_dir)
        )

    def _get_storage_dir(self) -> str:
        """Get hyperparameter storage directory from config."""
        try:
            model_cfg = self.config.core.model_testing_config
            if hasattr(model_cfg, 'hyperparameter_history_dir'):
                return model_cfg.hyperparameter_history_dir
            else:
                # Default fallback
                return "hyperparameter_storage"
        except AttributeError as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Could not find hyperparameter storage directory in config, using default",
                error=str(e)
            )
            return "hyperparameter_storage"

    def get_current_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get the current hyperparameters for a model.

        Priority:
        1. Current best from storage directory (if exists and optimization enabled)
        2. Baseline from config object (always available)

        Args:
            model_name: Name of the model

        Returns:
            Dict containing model parameters
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Getting current parameters",
            model_name=model_name
        )

        try:
            # Get model-specific config
            model_config = self._get_model_config(model_name)

            # Determine which parameter set to use
            use_baseline = True
            if model_config and hasattr(model_config, 'use_baseline_hyperparameters'):
                use_baseline = model_config.use_baseline_hyperparameters
            if model_config and hasattr(model_config, 'perform_hyperparameter_optimization'):
                if model_config.perform_hyperparameter_optimization:
                    use_baseline = False

            # Try to load current best from storage if not using baseline
            if not use_baseline:
                current_best = self._load_current_best(model_name)
                if current_best is not None:
                    self.app_logger.structured_log(
                        logging.INFO,
                        "Loaded current best parameters from storage",
                        model_name=model_name
                    )
                    return current_best
                else:
                    self.app_logger.structured_log(
                        logging.INFO,
                        "No current best found in storage, falling back to baseline",
                        model_name=model_name
                    )

            # Load baseline from config object
            params = self._get_baseline_from_config(model_name)

            if params is None:
                self.app_logger.structured_log(
                    logging.WARNING,
                    f"No baseline parameters found for {model_name}, returning empty dict"
                )
                return {}

            self.app_logger.structured_log(
                logging.INFO,
                "Loaded baseline parameters from config",
                model_name=model_name,
                param_count=len(params)
            )

            return params

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'hyperparameter_management',
                f"Error getting current parameters for {model_name}",
                original_error=str(e)
            )

    def _get_model_config(self, model_name: str) -> Any:
        """
        Get model-specific configuration from config object.

        Args:
            model_name: Name of the model (e.g., 'xgboost', 'lightgbm')

        Returns:
            Model-specific configuration object or None
        """
        try:
            # Remove underscores for config access (xgboost -> xgboost, sklearn_randomforest -> sklearnrandomforest)
            config_name = model_name.replace('_', '')

            # Try to access from models config
            if hasattr(self.config, 'models'):
                return getattr(self.config.models, config_name, None)

            # Try to access from core.models config
            if hasattr(self.config, 'core') and hasattr(self.config.core, 'models'):
                return getattr(self.config.core.models, config_name, None)

            return None

        except AttributeError:
            self.app_logger.structured_log(
                logging.WARNING,
                f"No configuration found for model: {model_name}"
            )
            return None

    def _get_baseline_from_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get baseline hyperparameters from the config object.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of baseline parameters or None if not found
        """
        try:
            # Access config.core.hyperparameters.{model_name}.baseline
            if not hasattr(self.config, 'core'):
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Config does not have 'core' attribute"
                )
                return None

            if not hasattr(self.config.core, 'hyperparameters'):
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Config does not have 'core.hyperparameters' attribute"
                )
                return None

            # Get the model's hyperparameters section
            model_hyperparams = getattr(self.config.core.hyperparameters, model_name, None)

            if model_hyperparams is None:
                self.app_logger.structured_log(
                    logging.WARNING,
                    f"No hyperparameters found for model: {model_name}"
                )
                return None

            # Get baseline config
            baseline = getattr(model_hyperparams, 'baseline', None)

            if baseline is None:
                self.app_logger.structured_log(
                    logging.WARNING,
                    f"No baseline hyperparameters found for model: {model_name}"
                )
                return None

            # Convert SimpleNamespace to dict
            if hasattr(baseline, '__dict__'):
                return vars(baseline)
            else:
                return baseline

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                f"Error loading baseline parameters for {model_name}",
                error=str(e)
            )
            return None

    def _load_current_best(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load current best parameters from storage directory.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of parameters or None if not found
        """
        try:
            current_best_file = self.current_best_dir / f"{model_name}.json"

            if not self.app_file_handler.join_paths(current_best_file).exists():
                return None

            data = self.app_file_handler.read_json(current_best_file)

            # Extract params from the data structure
            if "params" in data:
                return data["params"]
            else:
                # If no "params" key, assume entire file is parameters
                # but exclude metadata fields
                model_cfg = self.config.core.model_testing_config
                if hasattr(model_cfg, 'hyperparameter_metadata'):
                    metadata_fields = model_cfg.hyperparameter_metadata
                    return {k: v for k, v in data.items() if k not in metadata_fields}
                else:
                    return data

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                f"Error loading current best parameters for {model_name}",
                error=str(e)
            )
            return None

    def update_best_params(self, model_name: str, new_params: Dict[str, Any],
                          metrics: Dict[str, float], experiment_id: str,
                          run_id: Optional[str] = None, description: Optional[str] = None) -> None:
        """
        Update the best hyperparameters for a model.
        Saves to storage directory only, never modifies config files.

        Args:
            model_name: Name of the model
            new_params: New hyperparameters to save
            metrics: Performance metrics for the new parameters
            experiment_id: MLflow experiment ID
            run_id: Optional MLflow run ID
            description: Optional description of the parameter set
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Updating best parameters",
            model_name=model_name,
            metrics=metrics,
            run_id=run_id
        )

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

            # Update current best in storage
            self._update_current_best_storage(model_name, new_set)

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
        model_cfg = self.config.core.model_testing_config
        if hasattr(model_cfg, 'categorical_features'):
            categorical_features = model_cfg.categorical_features
        
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
        """Save hyperparameter set to history directory."""
        history_file = self.history_dir / f"{model_name}_history.json"

        try:
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
                "Saved parameter set to history",
                history_file=str(history_file),
                history_count=len(history)
            )

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                f"Error saving to history file",
                error=str(e),
                history_file=str(history_file)
            )

    def _update_current_best_storage(self, model_name: str, param_set: HyperparameterSet) -> None:
        """Update current best parameters in storage directory (NOT in configs)."""
        try:
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

            # Save to storage directory
            current_best_path = self.current_best_dir / f"{model_name}.json"
            self.app_file_handler.write_json(current_best, current_best_path)

            self.app_logger.structured_log(
                logging.INFO,
                "Updated current best parameters in storage",
                file_path=str(current_best_path)
            )

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                "Error updating current best parameters in storage",
                error=str(e),
                model_name=model_name
            )

    def _load_history(self, model_name: str) -> List[HyperparameterSet]:
        """Load hyperparameter history for a model."""
        history_file = self.history_dir / f"{model_name}_history.json"

        try:
            try:
                history_data = self.app_file_handler.read_json(history_file)
                return [HyperparameterSet.from_dict(item) for item in history_data]
            except FileNotFoundError:
                self.app_logger.structured_log(
                    logging.INFO,
                    "No history file found for model",
                    model_name=model_name,
                    history_file=str(history_file)
                )
                return []
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                "Error loading history for model",
                model_name=model_name,
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
