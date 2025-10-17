import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import optuna
from optuna.trial import Trial
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

from .base_hyperparams_optimizer import BaseHyperparamsOptimizer
from ..hyperparams_managers.base_hyperparams_manager import BaseHyperparamsManager

class OptunaOptimizer(BaseHyperparamsOptimizer):
    """Optuna-based hyperparameter optimizer."""
    
    def __init__(self,
                 config: BaseConfigManager,
                 hyperparameter_manager: BaseHyperparamsManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """Initialize Optuna optimizer with dependencies."""
        self.config = config
        self.hyperparameter_manager = hyperparameter_manager
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.study = None
        self.best_params = None
        
        self.app_logger.structured_log(
            logging.INFO, 
            "OptunaOptimizer initialized"
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    def _create_study(self, direction: str) -> None:
        """Create a new Optuna study."""
        try:
            self.study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
            )
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'optimization',
                "Failed to create Optuna study",
                error_message=str(e)
            )

    @log_performance
    def optimize(self,
                objective_func: Optional[Callable] = None,
                X = None,
                y = None,
                model_type: str = None,
                n_splits: int = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna and model_testing_config settings.

        Args:
            objective_func: Optional custom objective function
            X: Training data
            y: Target data
            model_type: Type of model to optimize
            n_splits: Number of cross-validation splits

        Returns:
            Dictionary of optimized parameters
        """
        try:
            # Local config aliases
            model_cfg = self.config.core.model_testing_config

            # Get parameter space and optimization settings from model config
            param_space, opt_settings = self._get_param_space(model_type)

            # Get optimization settings
            n_trials = opt_settings.get('n_trials', 100)
            scoring = 'auc'  # Default scoring
            direction = 'maximize'  # For AUC, we want to maximize
            n_splits = n_splits or model_cfg.n_splits
            cv_type = model_cfg.cross_validation_type

            self.app_logger.structured_log(
                logging.INFO,
                "Starting optimization",
                model_type=model_type,
                n_trials=n_trials,
                scoring=scoring,
                direction=direction
            )
            
            self._create_study(direction)
            
            # Choose appropriate optimization method based on model type
            if model_type.lower() == "xgboost":
                self.study = self._optimize_xgboost(X, y, param_space, n_trials, n_splits, cv_type, scoring)
            elif model_type.lower() == "lightgbm":
                self.study = self._optimize_lightgbm(X, y, param_space, n_trials, n_splits, cv_type, scoring)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Create unique run ID and merge parameters
            run_id = f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            final_best_params = self._merge_parameters(param_space, self.study.best_params)
            
            self.app_logger.structured_log(
                logging.INFO,
                "Optimization complete",
                model_type=model_type,
                best_value=self.study.best_value,
                run_id=run_id
            )
            
            # Update hyperparameter manager with new best parameters
            self.hyperparameter_manager.update_best_params(
                model_name=model_type,
                new_params=final_best_params,
                metrics={scoring: self.study.best_value},
                experiment_id="Optuna",
                run_id=run_id,
                description="Optuna optimization"
            )

            # Return the newly optimized parameters
            return final_best_params
        
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'optimization',
                "Error during optimization",
                error_message=str(e)
            )

    def _get_param_space(self, model_type: str):
        """Get parameter space and optimization settings from model-specific config.

        Returns:
            Tuple of (param_space_dict, optimization_settings_dict)
        """
        try:
            # Get model config name (e.g., xgboost -> xgboost_config)
            config_name = model_type.replace('_', '') + '_config'

            # Access model config
            if hasattr(self.config.core, 'models'):
                model_config = getattr(self.config.core.models, config_name, None)
                if model_config is None:
                    raise ValueError(f"No configuration found for {config_name}")

                # Extract optimization settings
                if hasattr(model_config, 'optimization'):
                    opt_config = model_config.optimization

                    # Build parameter space
                    param_space = {
                        'static_params': vars(model_config.static_params) if hasattr(model_config, 'static_params') else {},
                        'dynamic_params': vars(opt_config.param_space) if hasattr(opt_config, 'param_space') else {}
                    }

                    # Extract optimization settings
                    opt_settings = {
                        'n_trials': opt_config.n_trials if hasattr(opt_config, 'n_trials') else 100
                    }

                    return param_space, opt_settings
                else:
                    raise ValueError(f"No optimization config found in {config_name}")
            else:
                raise ValueError("config.core.models not found")

        except Exception as e:
            raise ValueError(f"Error loading parameter space for {model_type}: {str(e)}")

    def _namespace_to_dict(self, namespace: Any) -> Dict:
        """Recursively convert SimpleNamespace to dict."""
        if hasattr(namespace, '__dict__'):
            return {k: self._namespace_to_dict(v) for k, v in vars(namespace).items()}
        elif isinstance(namespace, (list, tuple)):
            return type(namespace)(self._namespace_to_dict(x) for x in namespace)
        elif isinstance(namespace, dict):
            return {k: self._namespace_to_dict(v) for k, v in namespace.items()}
        return namespace

    def _merge_parameters(self, param_space: Dict, best_params: Dict) -> Dict:
        """Merge static parameters with optimized parameters."""
        final_params = param_space.get('static_params', {}).copy()
        final_params.update(best_params)
        return final_params

    def _optimization_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback to log optimization progress."""
        self.app_logger.structured_log(
            logging.INFO,
            "Trial completed",
            trial_number=trial.number,
            value=trial.value,
            params=trial.params
        )

    def _process_parameters(self, trial: optuna.Trial, param_space: Dict) -> Dict:
        """Process both static and dynamic parameters from parameter space."""
        try:
            # Start with static parameters
            params = param_space.get('static_params', {}).copy()
            
            # Process dynamic parameters
            dynamic_params = param_space.get('dynamic_params', {})
            for param_name, config in dynamic_params.items():
                try:
                    if not isinstance(config, list):
                        params[param_name] = config
                        continue
                    
                    # Config format: [low, high, type, log(optional)]
                    param_type = config[2]
                    if param_type == "int":
                        params[param_name] = trial.suggest_int(param_name, config[0], config[1])
                    elif param_type == "float":
                        log = config[3] if len(config) > 3 else False
                        low = float(config[0])
                        high = float(config[1])
                        params[param_name] = trial.suggest_float(param_name, low, high, log=log)
                    elif param_type == "categorical":
                        params[param_name] = trial.suggest_categorical(param_name, config[0])
                    
                    self.app_logger.structured_log(
                        logging.DEBUG,
                        "Processed parameter",
                        param_name=param_name,
                        value=params[param_name],
                        param_type=param_type
                    )
                        
                except Exception as e:
                    raise self.error_handler.create_error_handler(
                        'optimization',
                        f"Error processing parameter {param_name}",
                        error_message=f"Config: {config}. Error: {str(e)}"
                    )
            
            return params
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'optimization',
                "Error processing parameters",
                error_message=str(e)
            )

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found during optimization."""
        if self.best_params is None:
            raise self.error_handler.create_error_handler(
                'optimization',
                "No optimization has been performed yet"
            )
        return self.best_params

    def _optimize_xgboost(self, X, y, param_space, n_trials, n_splits, cv_type, scoring):
        """Optimize XGBoost model hyperparameters."""
        # Get XGBoost-specific settings from model config
        xgb_config = self.config.core.models.xgboost_config
        enable_categorical = xgb_config.enable_categorical if hasattr(xgb_config, 'enable_categorical') else False
        num_boost_round = xgb_config.num_boost_round if hasattr(xgb_config, 'num_boost_round') else 100
        early_stopping_rounds = xgb_config.early_stopping_rounds if hasattr(xgb_config, 'early_stopping_rounds') else 10

        def objective(trial):
            params = self._process_parameters(trial, param_space)
            scores = []

            # Use appropriate cross-validation strategy
            if cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=n_splits)
            else:
                kf = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.config.random_state
                )

            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                dtrain = xgb.DMatrix(
                    X_train,
                    label=y_train,
                    enable_categorical=enable_categorical
                )
                dval = xgb.DMatrix(
                    X_val,
                    label=y_val,
                    enable_categorical=enable_categorical
                )

                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False
                )

                y_pred = model.predict(dval)
                scores.append(self._calculate_score(y_val, y_pred, scoring))

            return np.mean(scores)

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[self._optimization_callback]
        )

        return self.study

    def _optimize_lightgbm(self, X, y, param_space, n_trials, n_splits, cv_type, scoring):
        """Optimize LightGBM model hyperparameters."""
        # Local config alias
        model_cfg = self.config.core.model_testing_config

        def objective(trial):
            params = self._process_parameters(trial, param_space)
            scores = []

            # Use appropriate cross-validation strategy
            if cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=n_splits)
            else:
                kf = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.config.random_state
                )

            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                train_data = lgb.Dataset(
                    X_train,
                    label=y_train,
                    categorical_feature=model_cfg.categorical_features
                )
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val,
                    categorical_feature=model_cfg.categorical_features,
                    reference=train_data
                )
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=self.config.LightGBM.num_boost_round,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(self.config.LightGBM.early_stopping),
                        lgb.log_evaluation(0)
                    ]
                )
                
                y_pred = model.predict(X_val)
                scores.append(self._calculate_score(y_val, y_pred, scoring))
            
            return np.mean(scores)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[self._optimization_callback]
        )
        
        return self.study

    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
        """Calculate the score for a prediction using the specified metric."""
        from sklearn.metrics import roc_auc_score
        
        if scoring == 'auc':
            return roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")