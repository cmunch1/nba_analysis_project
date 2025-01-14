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

logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer.
    Uses configuration from model_testing_config for consistent settings.
    """
    
    @log_performance
    def __init__(self, config: AbstractConfig):
        self.config = config
        self.study = None
        self.best_params = None
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
    
    @log_performance
    def optimize(self, 
                objective_func: Optional[Callable] = None,
                param_space: Dict[str, Any] = None,
                X = None,
                y = None,
                model_type: str = None,
                cv: int = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna and model_testing_config settings.
        """
        # Get optimization settings from config
        n_trials = self.config.optuna.n_trials
        scoring = self.config.optuna.scoring
        direction = self.config.optuna.direction

        structured_log(logger, logging.INFO, 
                      "Starting optimization",
                      model_type=model_type,
                      n_trials=n_trials,
                      scoring=scoring,
                      direction=direction)
        
        try:
            self._create_study(direction)
            
            # Use cross-validation settings from config
            cv = cv or self.config.n_splits
            cv_type = self.config.cross_validation_type

            if model_type is not None:
                if model_type.lower() == "xgboost":
                    return self._optimize_xgboost(X, y, param_space, n_trials, cv, cv_type, scoring)
                elif model_type.lower() == "lgbm":
                    return self._optimize_lightgbm(X, y, param_space, n_trials, cv, cv_type, scoring)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            
            if objective_func is None:
                raise ValueError("Either model_type or objective_func must be provided")
            
            # Use custom objective function with provided param space
            def objective(trial: Trial) -> float:
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["values"])
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, param_config["low"], param_config["high"])
                    elif param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name, param_config["low"], param_config["high"],
                            log=param_config.get("log", False))
                
                return objective_func(params)
            
            self.study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=[self._optimization_callback]
            )
            
            self.best_params = self.study.best_params
            
            return {
                "best_params": self.best_params,
                "best_value": self.study.best_value,
                "n_trials": len(self.study.trials),
                "study": self.study
            }
            
        except Exception as e:
            raise OptimizationError("Error during optimization",
                                  error_message=str(e))
    
    @log_performance
    def _optimize_xgboost(self, X, y, param_space, n_trials, cv, cv_type, scoring):
        """Optimize XGBoost model hyperparameters using config settings."""
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'tree_method': self.config.XGB.tree_method if hasattr(self.config.XGB, 'tree_method') else 'hist',
                'device': self.config.XGB.device if hasattr(self.config.XGB, 'device') else 'cpu'
            }
            
            # Get parameters from param_space
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["values"])
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"])
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"],
                        log=param_config.get("log", False))
            
            scores = []
            
            # Use appropriate cross-validation strategy from config
            if cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=cv)
            else:  # Default to StratifiedKFold
                kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.config.random_state)
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                dtrain = xgb.DMatrix(X_train, label=y_train,
                                   enable_categorical=self.config.enable_categorical)
                dval = xgb.DMatrix(X_val, label=y_val,
                                 enable_categorical=self.config.enable_categorical)
                
                # Train model using config settings
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.config.XGB.num_boost_round,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=self.config.XGB.early_stopping_rounds,
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
        
        self.best_params = self.study.best_params
        
        # Train final model with best parameters
        best_params = {
            **self.best_params,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0,
            'tree_method': self.config.XGB.tree_method if hasattr(self.config.XGB, 'tree_method') else 'hist',
            'device': self.config.XGB.device if hasattr(self.config.XGB, 'device') else 'cpu'
        }
        
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=self.config.enable_categorical)
        best_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=self.config.XGB.num_boost_round
        )
        
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "best_model": best_model,
            "n_trials": len(self.study.trials),
            "study": self.study
        }
    
    @log_performance
    def _optimize_lightgbm(self, X, y, param_space, n_trials, cv, cv_type, scoring):
        """Optimize LightGBM model hyperparameters using config settings."""
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1
            }
            
            # Get parameters from param_space
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["values"])
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"])
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"],
                        log=param_config.get("log", False))
            
            scores = []
            
            # Use appropriate cross-validation strategy from config
            if cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=cv)
            else:  # Default to StratifiedKFold
                kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.config.random_state)
            
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
                    num_boost_round=self.config.LGBM.num_boost_round,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(self.config.LGBM.early_stopping),
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
        
        self.best_params = self.study.best_params
        
        # Train final model with best parameters
        best_params = {
            **self.best_params,
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1
        }
        
        train_data = lgb.Dataset(X, label=y, categorical_feature=self.config.categorical_features)
        best_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=self.config.LGBM.num_boost_round
        )
        
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "best_model": best_model,
            "n_trials": len(self.study.trials),
            "study": self.study
        }

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