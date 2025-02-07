from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, List, Callable
import optuna
from optuna.trial import Trial
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from ...config.config import AbstractConfig
from ...logging.logging_utils import log_performance, structured_log
from ...error_handling.custom_exceptions import OptimizationError
from ..abstract_model_testing import AbstractHyperparameterManager
from datetime import datetime
from types import SimpleNamespace


logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer.
    Uses configuration from model_testing_config for consistent settings.
    """
    
    @log_performance
    def __init__(self, config: AbstractConfig, hyperparameter_manager: AbstractHyperparameterManager):
        self.config = config
        self.study = None
        self.best_params = None
        self.param_manager = hyperparameter_manager
        structured_log(logger, logging.INFO, "OptunaOptimizer initialized")
    
    def _create_study(self, direction: str) -> None:
        """Create a new Optuna study."""
        try:
            self.study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
            )
        except Exception as e:
            raise OptimizationError("Failed to create Optuna study", 
                                  error_message=str(e))
    
    def _namespace_to_dict(self, namespace):
        """Recursively convert SimpleNamespace to dict"""
        if isinstance(namespace, SimpleNamespace):
            return {k: self._namespace_to_dict(v) for k, v in vars(namespace).items()}
        elif isinstance(namespace, (list, tuple)):
            return type(namespace)(self._namespace_to_dict(x) for x in namespace)
        elif isinstance(namespace, dict):
            return {k: self._namespace_to_dict(v) for k, v in namespace.items()}
        return namespace

    @log_performance
    def optimize(self, 
                objective_func: Optional[Callable] = None,
                X = None,
                y = None,
                model_type: str = None,
                n_splits: int = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna and model_testing_config settings.
        """
        # Get parameter space from config based on model type and convert to dict
        param_space = None
        if model_type:
            if model_type.lower() == "xgboost":
                param_space = self._namespace_to_dict(self.config.xgb_param_space)
            elif model_type.lower() == "lightgbm":
                param_space = self._namespace_to_dict(self.config.lgbm_param_space)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")



        # Get optimization settings from config
        n_trials = self.config.optuna.n_trials
        scoring = self.config.optuna.scoring
        direction = self.config.optuna.direction
        n_splits = n_splits or self.config.n_splits
        cv_type = self.config.cross_validation_type

        structured_log(logger, logging.INFO, 
                    "Starting optimization",
                    model_type=model_type,
                    n_trials=n_trials,
                    scoring=scoring,
                    direction=direction)
        
        try:
            self._create_study(direction)
            
            result = None
            if model_type.lower() == "xgboost":
                self.study = self._optimize_xgboost(X, y, param_space, n_trials, n_splits, cv_type, scoring)
            elif model_type.lower() == "lightgbm":
                self.study = self._optimize_lightgbm(X, y, param_space, n_trials, n_splits, cv_type, scoring)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
 
                

            run_id = f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Merge static parameters with best parameters from study
            final_best_params = param_space.get('static_params', {}).copy()
            final_best_params.update(self.study.best_params)
            
            structured_log(logger, logging.INFO, 
                        "Optimization complete, updating best parameters",
                        model_type=model_type,
                        best_value=self.study.best_value,
                        run_id=run_id)
            
            # Create a HyperparameterSet instance for the new parameters
            self.param_manager.update_best_params(
                model_name=model_type,
                new_params=final_best_params,  # Using merged parameters
                metrics={scoring: self.study.best_value},
                experiment_id="Optuna",
                run_id=run_id,
                description="Optuna optimization"
            )
            
            if self.config.always_use_new_hyperparameters:
                return final_best_params
            else:
                return self.param_manager.get_current_params(model_name=model_type)
        
        except Exception as e:
            raise OptimizationError("Error during optimization",
                                error_message=str(e))
    
    @log_performance
    def _optimize_xgboost(self, X, y, param_space, n_trials, n_splits, cv_type, scoring):
        """Optimize XGBoost model hyperparameters using config settings."""
        def objective(trial):
            # Process parameters using the new method
            params = self._process_parameters(trial, param_space)
            
            structured_log(logger, logging.INFO, 
                          "Final parameters for trial", 
                          trial_number=trial.number, 
                          params=params)

            scores = []
            
            # Use appropriate cross-validation strategy from config
            if cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=n_splits)
            else:  
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                dtrain = xgb.DMatrix(X_train, label=y_train,
                                   enable_categorical=self.config.XGBoost.enable_categorical)
                dval = xgb.DMatrix(X_val, label=y_val,
                                 enable_categorical=self.config.XGBoost.enable_categorical)
                

                # Train model using config settings
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.config.XGBoost.num_boost_round,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=self.config.XGBoost.early_stopping_rounds,
                    verbose_eval=False

                )
                
                # Get predictions
                y_pred = model.predict(dval)
                
                # Calculate score
                if scoring == 'auc':
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y_val, y_pred)
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring}")
                
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[self._optimization_callback]
        )
               
        return self.study
    
    @log_performance
    def _optimize_lightgbm(self, X, y, param_space, n_trials, n_splits, cv_type, scoring):
        """Optimize LightGBM model hyperparameters using config settings."""
        def objective(trial):
            # Process parameters using the new method
            params = self._process_parameters(trial, param_space)
            
            structured_log(logger, logging.INFO, 
                          "Final parameters for trial", 
                          trial_number=trial.number, 
                          params=params)

            scores = []
            
            # Use appropriate cross-validation strategy from config
            if cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=n_splits)
            else:  # Default to StratifiedKFold
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train,
                                       categorical_feature=self.config.categorical_features)
                val_data = lgb.Dataset(X_val, label=y_val,
                                     categorical_feature=self.config.categorical_features,
                                     reference=train_data)
                
                # Train model using config settings
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
                
                # Get predictions
                y_pred = model.predict(X_val)
                
                # Calculate score
                if scoring == 'auc':
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y_val, y_pred)
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring}")
                
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[self._optimization_callback]
        )
        
        return self.study

    def _optimization_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback to log optimization progress."""
        structured_log(logger, logging.INFO,
                      "Trial completed",
                      trial_number=trial.number,
                      value=trial.value,
                      params=trial.params)
    
    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found during optimization."""
        if self.best_params is None:
            raise OptimizationError("No optimization has been performed yet")
        return self.best_params

    def _process_parameters(self, trial, param_space: dict) -> dict:
        """
        Process both static and dynamic parameters from parameter space.
        
        Args:
            trial: Optuna trial object
            param_space: Dictionary containing parameters configuration
                Format matches config YAML structure with static_params and dynamic_params
        Returns:
            dict: Combined parameters dictionary for model training
        """
        # Start with static parameters if they exist
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
                    # Convert string values to float if necessary
                    low = float(config[0])
                    high = float(config[1])
                    params[param_name] = trial.suggest_float(param_name, low, high, log=log)
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, config[0])
                
                structured_log(logger, logging.DEBUG,
                             "Processed parameter",
                             param_name=param_name,
                             value=params[param_name],
                             param_type=param_type)
                    
            except Exception as e:
                raise OptimizationError(
                    f"Error processing parameter {param_name}",
                    error_message=f"Config: {config}. Error: {str(e)}"
                )
        
        return params