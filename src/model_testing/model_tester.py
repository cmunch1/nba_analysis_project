import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Union, Any
import yaml
import os
from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
from ..config.config import AbstractConfig
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import ModelTestingError
from .abstract_model_testing import AbstractModelTester
from ..data_access.data_access import AbstractDataAccess
from .data_classes import ModelTrainingResults, ClassificationMetrics, PreprocessingResults
from .modular_preprocessor import ModularPreprocessor
import lightgbm as lgb
from .hyperparameter_manager import HyperparameterManager

logger = logging.getLogger(__name__)

class ModelTester(AbstractModelTester):
    @log_performance
    def __init__(self, config: AbstractConfig, hyperparameter_manager: HyperparameterManager):
        """
        Initialize the ModelTester class.

        Args:
            config (AbstractConfig): Configuration object containing model testing parameters.
            hyperparameter_manager (HyperparameterManager): Manager for model hyperparameters.
        """
        self.config = config
        self.hyperparameter_manager = hyperparameter_manager
        self.preprocessor = ModularPreprocessor(config)
        structured_log(logger, logging.INFO, "ModelTester initialized",
                      config_type=type(config).__name__)

    def get_model_params(self, model_name: str) -> Dict:
        """
        Get the current best hyperparameters for a model from the configuration.

        Args:
            model_name (str): Name of the model to get parameters for.

        Returns:
            Dict: Dictionary containing the model's hyperparameters.

        Raises:
            ModelTestingError: If there's an error retrieving the hyperparameters.
        """
        try:
            return self.hyperparameter_manager.get_current_params(model_name)
        except Exception as e:
            raise ModelTestingError(f"Error getting hyperparameters for {model_name}",
                                  error_message=str(e))

    @log_performance
    def prepare_data(self, df: pd.DataFrame, model_name: str = None, is_training: bool = True, preprocessing_results: PreprocessingResults = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the data for model training or validation with preprocessing.

        Args:
            df (pd.DataFrame): Input dataframe.
            model_name (str): Name of the model to prepare data for.
            is_training (bool): Whether this is training data (fit_transform) or validation data (transform)

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Tuple containing the feature dataframe and the target series.
        """
        structured_log(logger, logging.INFO, "Starting data preparation",
                    input_shape=df.shape,
                    is_training=is_training)
        
        primary_ids = None

        try:

            # sort chronologically in case TimeSeriesSplit is used for cross-validation
            df = df.sort_values(by=self.config.sort_columns, ascending=self.config.sort_order)

            target = self.config.home_team_prefix + self.config.target_column
            y = df[target]
            X = df.drop(columns=[target])

            if self.config.primary_id_column:
                primary_ids = X[self.config.primary_id_column]
                X = X.drop(columns=[self.config.primary_id_column])

            if self.config.non_useful_columns:
                X = X.drop(columns=self.config.non_useful_columns)

            if self.config.perform_preprocessing:
                # Store original column names
                original_columns = X.columns.tolist()

                # explicitly convert categorical features to category type
                for col in self.config.categorical_features:
                    X[col] = X[col].astype('category')

                # Apply preprocessing with tracking
                if is_training:
                    X, preprocessing_results = self.preprocessor.fit_transform(
                        X,
                        y=y,
                        model_name=model_name,
                        preprocessing_results=preprocessing_results
                    )
                else:
                    X = self.preprocessor.transform(X)
                
                # Ensure column names are preserved after preprocessing
                if isinstance(X, pd.DataFrame):
                    if len(X.columns) == len(original_columns):
                        X.columns = original_columns
                else:
                    X = pd.DataFrame(X, columns=original_columns)

                if is_training:
                    # Convert preprocessing results to serializable format before logging
                    steps_summary = [
                        {
                            'name': step.name,
                            'type': step.type,
                            'n_columns': len(step.columns)
                        }
                        for step in preprocessing_results.steps
                    ]
                    
                    structured_log(logger, logging.INFO, "Preprocessing results",
                                steps_overview=steps_summary)

            X = self._reduce_memory_footprint(X)
            
            structured_log(logger, logging.INFO, "Data preparation completed",
                        output_shape=X.shape)
            return X, y, preprocessing_results, primary_ids
        
        except Exception as e:
            raise ModelTestingError("Error in data preparation",
                                error_message=str(e),
                                dataframe_shape=df.shape)
        
    def perform_oof_cross_validation(self, X: pd.DataFrame, y: pd.Series, model_name: str, model_params: Dict, full_results: ModelTrainingResults) -> ModelTrainingResults:
        """
        Perform Out-of-Fold (OOF) cross-validation with proper handling of TimeSeriesSplit results.
        """
        structured_log(logger, logging.INFO, f"{model_name} - Starting OOF cross-validation",
                    input_shape=X.shape)
        try:
            # Initialize predictions and SHAP arrays with NaN
            full_results.predictions = np.full(len(y), np.nan)
            if self.config.calculate_shap_values:
                full_results.shap_values = np.full((len(y), X.shape[1]), np.nan)
                if self.config.calculate_shap_interactions:
                    full_results.shap_interaction_values = np.full((len(y), X.shape[1], X.shape[1]), np.nan)
            
            if self.config.cross_validation_type == "StratifiedKFold":
                kf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)
            elif self.config.cross_validation_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=self.config.n_splits)
            else:
                raise ValueError(f"Unsupported CV type: {self.config.cross_validation_type}")

            feature_importance_accumulator = np.zeros(X.shape[1])
            n_folds_with_importance = 0
            processed_samples = np.zeros(len(y), dtype=bool)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                fold_results = ModelTrainingResults(X_train.shape)
                oof_results = self._train_model(
                    X_train, y_train,
                    X_val, y_val,
                    fold,
                    model_name, model_params,
                    fold_results

                )
                
                # Copy first fold's configuration
                if fold == 1:
                    full_results.num_boost_round = oof_results.num_boost_round
                    full_results.early_stopping = oof_results.early_stopping
                    full_results.enable_categorical = oof_results.enable_categorical
                    full_results.categorical_features = oof_results.categorical_features
                    full_results.model = oof_results.model
                    full_results.feature_names = oof_results.feature_names

                full_results.predictions[val_idx] = oof_results.predictions
                processed_samples[val_idx] = True

                if oof_results.shap_values is not None and self.config.calculate_shap_values:
                    full_results.shap_values[val_idx] = oof_results.shap_values

                    if (oof_results.shap_interaction_values is not None and 
                        self.config.calculate_shap_interactions):
                        full_results.shap_interaction_values[val_idx] = (
                            oof_results.shap_interaction_values
                        )

                if oof_results.feature_importance_scores is not None:
                    feature_importance_accumulator += oof_results.feature_importance_scores
                    n_folds_with_importance += 1

                if oof_results.learning_curve_data['raw_data']:
                    full_results.learning_curve_data['raw_data'].extend(
                        oof_results.learning_curve_data['raw_data']
                    )

                structured_log(logger, logging.INFO, f"Fold {fold} completed",
                            samples_processed=np.sum(processed_samples),
                            total_samples=len(y))

            if n_folds_with_importance > 0:
                full_results.feature_importance_scores = (
                    feature_importance_accumulator / n_folds_with_importance
                )

            # Handle TimeSeriesSplit unprocessed samples
            if not np.all(processed_samples):
                unprocessed_count = np.sum(~processed_samples)
                structured_log(logger, logging.WARNING,
                            "Some samples were not processed in cross-validation",
                            unprocessed_count=unprocessed_count,
                            cv_type=self.config.cross_validation_type)
                
                # Keep only processed samples in the results
                processed_mask = processed_samples
                if full_results.feature_data is None:
                    full_results.feature_data = X
                    full_results.target_data = y
                    
                full_results.feature_data = X[processed_mask]
                full_results.target_data = y[processed_mask]
                full_results.predictions = full_results.predictions[processed_mask]
                
                if full_results.shap_values is not None:
                    structured_log(logger, logging.INFO, "Filtering SHAP values for processed samples")
                    full_results.shap_values = full_results.shap_values[processed_mask]
                    if full_results.shap_interaction_values is not None:
                        full_results.shap_interaction_values = full_results.shap_interaction_values[processed_mask]

            structured_log(logger, logging.INFO, "Cross-validation completed",
                        final_feature_shape=full_results.feature_data.shape,
                        final_shap_shape=full_results.shap_values.shape if full_results.shap_values is not None else None)

            return full_results

        except Exception as e:
            raise ModelTestingError(
                "Error in OOF cross-validation",
                error_message=str(e),
                dataframe_shape=X.shape
            )
    
    @log_performance
    def perform_validation_set_testing(self, X: pd.DataFrame, y: pd.Series, 
                                    X_val: pd.DataFrame, y_val: pd.Series, 
                                    model_name: str, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """
        Perform model training and validation on a separate validation set.
        """
        try:
            fold = 1 # needs a value for consistency with OOF cross-validation
            
            results = self._train_model(
                X, y,
                X_val, y_val, fold,
                model_name, model_params,
                results
            )

            
            return results
            
        except Exception as e:
            raise ModelTestingError("Error in validation set testing",
                                error_message=str(e),
                                dataframe_shape=X.shape)

    @log_performance
    def calculate_classification_evaluation_metrics(self, y_true, y_prob, metrics: ClassificationMetrics) -> ClassificationMetrics:
        """Calculate classification metrics using probability scores and optimal threshold."""
        
        structured_log(logger, logging.INFO, "Calculating classification metrics",
                    n_samples=len(y_true))
        
        try:

            if np.any(np.isnan(y_prob)):
                structured_log(logger, logging.WARNING, 
                            "Unexpected NaN values found in predictions after OOF filtering",
                            nan_count=np.sum(np.isnan(y_prob)))
                raise ValueError("Unexpected NaN values in predictions after OOF filtering")
            
            # Find optimal threshold using ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Convert probabilities to predictions using optimal threshold
            y_pred = (y_prob >= optimal_threshold).astype(int)
            
            # Calculate metrics
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred)
            metrics.recall = recall_score(y_true, y_pred)
            metrics.f1 = f1_score(y_true, y_pred)
            metrics.auc = roc_auc_score(y_true, y_prob)
            metrics.optimal_threshold = float(optimal_threshold)
            
            # Add information about samples
            metrics.valid_samples = len(y_true)
            metrics.total_samples = len(y_true)  # These are now equal since filtering happened earlier
            metrics.nan_percentage = 0  # No NaNs should be present
            
            structured_log(logger, logging.INFO, "Classification metrics calculated",
                        optimal_threshold=float(optimal_threshold),
                        accuracy=float(metrics.accuracy),
                        auc=float(metrics.auc),
                        n_samples=len(y_true))
            
            return metrics
            
        except Exception as e:
            raise ModelTestingError("Error calculating classification metrics",
                                error_message=str(e),
                                n_samples=len(y_true) if y_true is not None else None,
                                n_predictions=len(y_prob) if y_prob is not None else None)
    
    @log_performance
    def _train_model(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, fold: int,
                     model_name: str, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a model on the given data."""
        structured_log(logger, logging.INFO, f"{model_name} - Starting model training",
                      input_shape=X.shape)
        try:
            
            # Debug column alignment
            structured_log(logger, logging.DEBUG, "Column comparison",
                        train_columns=X.columns.tolist(),
                        val_columns=X_val.columns.tolist())

            # Instead of using column indexing, ensure columns match using merge/join
            missing_cols = set(X.columns) - set(X_val.columns)
            extra_cols = set(X_val.columns) - set(X.columns)
            
            if missing_cols or extra_cols:
                structured_log(logger, logging.WARNING, "Column mismatch detected",
                            missing_columns=list(missing_cols),
                            extra_columns=list(extra_cols))
                
            inf_cols = X.columns[np.isinf(X).any()].tolist()
            structured_log(logger, logging.WARNING, "X Columns with inf values",
                        columns=inf_cols)
            inf_cols = X_val.columns[np.isinf(X_val).any()].tolist()
            structured_log(logger, logging.WARNING, "X_val Columns with inf values",
                        columns=inf_cols)
            
            results.model_name = model_name
            results.model_params = model_params           
            results.feature_names = X.columns.tolist()
            results.update_feature_data(X_val, y_val)
            
            match model_name:
                case "XGBoost":
                    results = self._train_XGBoost(X, y, X_val, y_val, fold, model_params, results)
                case "LGBM":
                    results = self._train_LGBM(X, y, X_val, y_val, fold,model_params, results)
                case _:
                    results = self._train_sklearn_model(X, y, X_val, y_val, fold, model_name, model_params, results)
            
            if results.predictions is None:
                raise ModelTestingError("Model training completed but predictions are None")
                                    
            structured_log(logger, logging.INFO, f"{model_name} - Model training completed",
                         predictions_shape=results.predictions.shape)
            
            return results
        
        except Exception as e:
            raise ModelTestingError("Error in model training",
                                  error_message=str(e),
                                  model_name=model_name,
                                  dataframe_shape=X.shape)
        
    @log_performance
    def _train_sklearn_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, fold: int, model_name: str, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """
        Train a scikit-learn model.

        Args:
            X_train (pd.DataFrame): Training feature dataframe.
            y_train (pd.Series): Training target series.
            X_val (pd.DataFrame): Validation feature dataframe.
            y_val (pd.Series): Validation target series.
            model_name (str): Name of the sklearn model (from tree or ensemble modules).
            model_params (Dict): Model hyperparameters.
            

        Returns:
            ModelTrainingResults: Object containing model, predictions, and various model analysis results.

        Raises:
            ModelTestingError: If there's an error during model training.
            ValueError: If the model name is not supported.
        """

        # Ensure X_val has the same columns in the same order as X_train
        X_val = X_val[X_train.columns]

        structured_log(logger, logging.INFO, f"{model_name} - Starting model training",
                       input_shape=X_train.shape)
        try:
            # Get the appropriate model class
            if hasattr(tree, model_name):
                model_class = getattr(tree, model_name)
            elif hasattr(ensemble, model_name):
                model_class = getattr(ensemble, model_name)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Initialize and train the model
            model = model_class()
            model.set_params(**model_params)
            model.fit(X_train, y_train)
            
            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)

            # Safely get feature importance scores
            try:
                if hasattr(model, 'feature_importances_'):
                    results.feature_importance_scores = model.feature_importances_
                    results.feature_names = X_train.columns.tolist()
                    structured_log(logger, logging.INFO, "Feature importance scores calculated successfully",
                                num_features=len(results.feature_importance_scores))
            except Exception as e:
                structured_log(logger, logging.WARNING, "Failed to get feature importance scores",
                            error=str(e),
                            error_type=type(e).__name__)
                results.feature_importance_scores = None
                results.feature_names = None

            # Note: Most sklearn models don't support SHAP values directly
            results.shap_values = None
            results.shap_interaction_values = None
            
            structured_log(logger, logging.INFO, f"{model_name} - Predictions generated",
                          predictions_type=type(results.predictions).__name__,
                          predictions_shape=results.predictions.shape,
                          predictions_sample=results.predictions[:5].tolist())
            
            return results
        
        except Exception as e:
            raise ModelTestingError("Error in sklearn model training",
                                     error_message=str(e),
                                     dataframe_shape=X_train.shape)
        
    @log_performance
    def _train_XGBoost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, fold: int,
                    model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train an XGBoost model"""
        structured_log(logger, logging.INFO, "Starting XGBoost training", 
                    input_shape=X_train.shape)
        
        try:
            # Create DMatrix objects
            dtrain = xgb.DMatrix(
                X_train, 
                label=y_train,
                feature_names=X_train.columns.tolist(),
                enable_categorical=self.config.enable_categorical
            )
            dval = xgb.DMatrix(
                X_val, 
                label=y_val,
                feature_names=X_val.columns.tolist(),
                enable_categorical=self.config.enable_categorical
            )

            # Train the model
            model = xgb.train(
                params=model_params,
                dtrain=dtrain,
                num_boost_round=self.config.XGB.num_boost_round,
                early_stopping_rounds=self.config.XGB.early_stopping_rounds,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                verbose_eval=self.config.XGB.verbose_eval
            )

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(dval)
            results.num_boost_round = self.config.XGB.num_boost_round
            results.early_stopping = self.config.XGB.early_stopping_rounds
            results.enable_categorical = self.config.enable_categorical
            results.categorical_features = self.config.categorical_features

            structured_log(logger, logging.INFO, "Generated predictions", 
                        predictions_shape=results.predictions.shape,
                        predictions_mean=float(np.mean(results.predictions)))

            # Calculate feature importance and store feature names
            try:
                importance_dict = model.get_score(importance_type='gain')
                results.feature_importance_scores = np.array([
                    importance_dict.get(feature, 0) 
                    for feature in X_val.columns
                ])
                results.feature_names = X_val.columns.tolist()
                
                structured_log(logger, logging.INFO, "Calculated feature importance",
                            num_features_with_importance=len(importance_dict))
                
            except Exception as e:
                structured_log(logger, logging.WARNING, "Failed to calculate feature importance",
                            error=str(e))
                results.feature_importance_scores = np.zeros(X_val.shape[1])

            # Calculate SHAP values if configured
            if self.config.calculate_shap_values:
                try:
                    # Calculate SHAP values
                    shap_values = model.predict(dval, pred_contribs=True)
                    
                    structured_log(logger, logging.INFO, "SHAP value details",
                                shap_values_shape=shap_values.shape,
                                shap_values_nulls=np.sum(np.isnan(shap_values)),
                                X_val_shape=X_val.shape)
                    
                    # Remove the bias term (last column) from SHAP values
                    if shap_values.shape[1] == X_val.shape[1] + 1:
                        structured_log(logger, logging.INFO, 
                                    "Removing bias term from SHAP values",
                                    original_shape=shap_values.shape)
                        shap_values = shap_values[:, :-1]
                        structured_log(logger, logging.INFO, 
                                    "SHAP values after bias removal",
                                    new_shape=shap_values.shape)
                    
                    # Store feature data and SHAP values
                    results.feature_data = X_val
                    results.target_data = y_val
                    results.shap_values = shap_values
                    
                    # Calculate interactions if configured
                    if self.config.calculate_shap_interactions:
                        n_features = X_val.shape[1]
                        estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)
                        
                        if estimated_memory <= self.config.max_shap_interaction_memory_gb:
                            interaction_values = model.predict(dval, pred_interactions=True)
                            # Remove bias term from interaction values if present
                            if interaction_values.shape[1] == X_val.shape[1] + 1:
                                interaction_values = interaction_values[:, :-1, :-1]
                            results.shap_interaction_values = interaction_values
                            structured_log(logger, logging.INFO, "Calculated SHAP interaction values",
                                        interaction_shape=interaction_values.shape)
                                
                except Exception as e:
                    structured_log(logger, logging.ERROR, "Failed to calculate SHAP values",
                                error=str(e),
                                error_type=type(e).__name__)
                    raise

            # Final validation of shapes
            if results.shap_values is not None:
                structured_log(logger, logging.INFO, "Final shape validation",
                            feature_data_shape=results.feature_data.shape,
                            shap_values_shape=results.shap_values.shape,
                            equal_shapes=results.feature_data.shape[1] == results.shap_values.shape[1])
                
            # Generate learning curve data if configured
            if self.config.generate_learning_curve_data:
                results = self._generate_XGB_learning_curve_data(X_train, y_train, X_val, y_val, fold, model_params, results)
            

            structured_log(logger, logging.INFO, "XGBoost training completed successfully",
                        predictions_shape=results.predictions.shape)
            


            return results

        except Exception as e:
            raise ModelTestingError(
                "Error in XGBoost model training",
                error_message=str(e),
                dataframe_shape=X_train.shape
            )   
         
    @log_performance
    def _train_LGBM(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """
        Train a LightGBM model with the enhanced ModelTrainingResults.
        Returns:
            ModelTrainingResults object with trained model and predictions
        """
        structured_log(logger, logging.INFO, "Starting LightGBM training", 
                    input_shape=X_train.shape)
        
        try:
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.config.categorical_features)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=self.config.categorical_features, reference=train_data)
            
            # Train model
            model = lgb.train(
                model_params,  
                train_data,
                num_boost_round=self.config.LGBM.num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(self.config.LGBM.early_stopping),
                    lgb.log_evaluation(self.config.LGBM.log_evaluation)
                ]
            )

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(X_val)
            results.num_boost_round = self.config.LGBM.num_boost_round
            results.early_stopping = self.config.LGBM.early_stopping
            results.enable_categorical = True  # LightGBM always enables categorical
            results.categorical_features = self.config.categorical_features

            structured_log(logger, logging.INFO, "Generated predictions", 
                        predictions_shape=results.predictions.shape,
                        predictions_mean=float(np.mean(results.predictions)))

            # Calculate feature importance
            try:
                results.feature_importance_scores = model.feature_importance('gain')
                results.feature_names = X_train.columns.tolist()
                
                structured_log(logger, logging.INFO, "Calculated feature importance",
                            num_features_with_importance=len(results.feature_importance_scores))
                
            except Exception as e:
                structured_log(logger, logging.WARNING, "Failed to calculate feature importance",
                            error=str(e))
                results.feature_importance_scores = np.zeros(X_train.shape[1])

            # Calculate SHAP values if configured
            if self.config.calculate_shap_values:
                try:
                    # Calculate SHAP values
                    shap_values = model.predict(X_val, pred_contrib=True)
                    
                    structured_log(logger, logging.INFO, "SHAP value details",
                                shap_values_shape=shap_values.shape,
                                shap_values_nulls=np.sum(np.isnan(shap_values)),
                                X_val_shape=X_val.shape)
                    
                    # Remove the bias term (last column) from SHAP values
                    if shap_values.shape[1] == X_val.shape[1] + 1:
                        structured_log(logger, logging.INFO, 
                                    "Removing bias term from SHAP values",
                                    original_shape=shap_values.shape)
                        shap_values = shap_values[:, :-1]
                        structured_log(logger, logging.INFO, 
                                    "SHAP values after bias removal",
                                    new_shape=shap_values.shape)
                    
                    # Store feature data and SHAP values
                    results.feature_data = X_val
                    results.target_data = y_val
                    results.shap_values = shap_values
                    
                    # Calculate interactions if configured
                    if self.config.calculate_shap_interactions:
                        n_features = X_val.shape[1]
                        estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)
                        
                        if estimated_memory <= self.config.max_shap_interaction_memory_gb:
                            interaction_values = model.predict(X_val, pred_interactions=True)
                            # Remove bias term from interaction values if present
                            if interaction_values.shape[1] == X_val.shape[1] + 1:
                                interaction_values = interaction_values[:, :-1, :-1]
                            results.shap_interaction_values = interaction_values
                            structured_log(logger, logging.INFO, "Calculated SHAP interaction values",
                                        interaction_shape=interaction_values.shape)
                            
                except Exception as e:
                    structured_log(logger, logging.ERROR, "Failed to calculate SHAP values",
                                error=str(e),
                                error_type=type(e).__name__)
                    raise

            # Final validation of shapes
            if results.shap_values is not None:
                structured_log(logger, logging.INFO, "Final shape validation",
                            feature_data_shape=results.feature_data.shape,
                            shap_values_shape=results.shap_values.shape,
                            equal_shapes=results.feature_data.shape[1] == results.shap_values.shape[1])
                
            # Generate learning curve data if configured
            if self.config.generate_learning_curve_data:
                results = self._generate_LGBM_learning_curve_data(X_train, y_train, X_val, y_val, fold, model_params, results)
        
            return results

        except Exception as e:

            structured_log(logger, logging.ERROR, "Error in LightGBM training",
                        error_message=str(e),
                        error_type=type(e).__name__,
                        dataframe_shape=X_train.shape)
            raise ModelTestingError(
                "Error in LightGBM model training",
                error_message=str(e),
                dataframe_shape=X_train.shape
            )
        
    @log_performance
    def _reduce_memory_footprint(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The primarily reduces the bitsize of ints and floats to reduce memory usage and improve performance.

        Args:
            df (pd.DataFrame): Input dataframe.
            date_column (str): Name of the date column. Defaults to 'date'.

        Returns:
            pd.DataFrame: Dataframe with optimized data types.
        """
        structured_log(logger, logging.INFO, "Starting memory footprint reduction",
                       input_shape=df.shape, input_memory=df.memory_usage().sum() / 1e6)
        try:

            # Optimize numeric columns
            for col in df.select_dtypes(include=['int', 'float']).columns:
                col_min, col_max = df[col].min(), df[col].max()

                # For integer columns
                if df[col].dtype.kind in ['i', 'u']:
                    if col_min >= -128 and col_max <= 127:
                        df[col] = df[col].astype(np.int8)
                    elif col_min >= -32768 and col_max <= 32767:
                        df[col] = df[col].astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

                # For float columns
                else:
                    df[col] = df[col].astype(np.float16)

            structured_log(logger, logging.INFO, "Memory footprint reduction completed",
                           output_shape=df.shape, output_memory=df.memory_usage().sum() / 1e6)
            return df
        except Exception as e:
            raise ModelTestingError("Error in memory footprint reduction",
                                     error_message=str(e),
                                     dataframe_shape=df.shape)

    def _debug_time_series_split(self, X, y, n_splits=5):
        """
        Debug function to analyze TimeSeriesSplit behavior and identify prediction gaps.
        """
        from sklearn.model_selection import TimeSeriesSplit
        import numpy as np
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Create an array to track which indices are used in validation
        coverage = np.zeros(len(X))
        fold_details = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            coverage[val_idx] += 1
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'val_start': val_idx[0],
                'val_end': val_idx[-1]
            })
        
        # Analyze coverage
        unused_indices = np.where(coverage == 0)[0]
        multiple_used = np.where(coverage > 1)[0]
        
        results = {
            'total_samples': len(X),
            'unused_count': len(unused_indices),
            'unused_indices': unused_indices,
            'unused_percentage': (len(unused_indices) / len(X)) * 100,
            'multiple_predictions_count': len(multiple_used),
            'fold_details': fold_details
        }
        
        return results

    # Helper function to analyze model predictions
    def _analyze_predictions(self, predictions, y):
        """
        Analyze prediction patterns and their relationship with the target variable.
        """
        nan_mask = np.isnan(predictions)
        results = {
            'total_samples': len(predictions),
            'nan_count': np.sum(nan_mask),
            'nan_percentage': (np.sum(nan_mask) / len(predictions)) * 100,
        }
        
        if np.any(~nan_mask):  # If there are any non-NaN predictions
            results.update({
                'non_nan_mean': np.mean(predictions[~nan_mask]),
                'non_nan_std': np.std(predictions[~nan_mask]),
            })
        
        return results
    

    def _generate_XGB_learning_curve_data(self, X_train, y_train, X_val, y_val, fold : int, model_params, 
                                        results) -> ModelTrainingResults:
        """Generate learning curve data for XGBoost with memory optimization."""

        structured_log(logger, logging.INFO, 
                    "Starting XGBoost learning curve generation",
                    fold=fold)

        # Calculate absolute sizes instead of percentages

        min_samples = max(int(len(X_train) * self.config.learning_curve.min_size), 
                        self.config.learning_curve.min_absolute_samples)
        max_samples = min(int(len(X_train) * self.config.learning_curve.max_size),
                        len(X_train))
        train_sizes = np.linspace(min_samples, max_samples,
                                self.config.learning_curve.n_points, dtype=int)

        # Set random seed for reproducibility
        np.random.seed(self.config.random_state + fold)
        
        # Create validation DMatrix once
        dval = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=X_val.columns.tolist(),
            enable_categorical=self.config.enable_categorical
        )

        # Adjust early stopping for smaller datasets
        min_early_stopping = max(5, self.config.XGB.early_stopping_rounds // 4)
        
        for n_samples in train_sizes:
            try:
                # Stratified sampling to maintain class distribution
                indices = self._stratified_sample_indices(y_train, n_samples)
                X_subset = X_train.iloc[indices]
                y_subset = y_train.iloc[indices]

                dtrain_subset = xgb.DMatrix(
                    X_subset,
                    label=y_subset,
                    feature_names=X_subset.columns.tolist(),
                    enable_categorical=self.config.enable_categorical
                )

                # Adjust early stopping based on sample size
                early_stopping = max(
                    min_early_stopping,
                    int(self.config.XGB.early_stopping_rounds * (n_samples / len(X_train)))
                )

                subset_model = xgb.train(
                    params=model_params,
                    dtrain=dtrain_subset,
                    num_boost_round=self.config.XGB.num_boost_round,
                    early_stopping_rounds=early_stopping,
                    evals=[(dtrain_subset, 'train'), (dval, 'eval')],
                    verbose_eval=self.config.XGB.verbose_eval
                )

                train_pred = subset_model.predict(dtrain_subset)
                val_pred = subset_model.predict(dval)

                # Calculate metrics
                train_score = self._calculate_model_score(y_subset, train_pred)
                val_score = self._calculate_model_score(y_val, val_pred)

                results.add_learning_curve_point(
                    train_size=n_samples,
                    train_score=train_score,
                    val_score=val_score,
                    fold=fold
                )

                del subset_model, dtrain_subset
                
            except Exception as e:
                structured_log(logger, logging.ERROR,
                            f"Error in XGBoost learning curve at size {n_samples}",
                            error=str(e))
                raise

        return results

    def _generate_LGBM_learning_curve_data(self, X_train, y_train, X_val, y_val, fold : int, model_params,
                                        results) -> ModelTrainingResults:
        """Generate learning curve data for LightGBM with memory optimization."""

        structured_log(logger, logging.INFO, 
                    "Starting LightGBM learning curve generation",
                    fold=fold)
        
        # Calculate absolute sizes instead of percentages
        min_samples = max(int(len(X_train) * self.config.learning_curve.min_size), 
                        self.config.learning_curve.min_absolute_samples)

        max_samples = min(int(len(X_train) * self.config.learning_curve.max_size),
                        len(X_train))
        train_sizes = np.linspace(min_samples, max_samples,
                                self.config.learning_curve.n_points, dtype=int)

        # Set random seed for reproducibility
        np.random.seed(self.config.random_state + fold)

        # Create validation dataset once
        val_data = lgb.Dataset(
            X_val, 
            label=y_val,
            categorical_feature=self.config.categorical_features,
            free_raw_data=False  # Keep raw data for reuse
        )

        # Adjust early stopping for smaller datasets
        min_early_stopping = max(5, self.config.LGBM.early_stopping // 4)

        for n_samples in train_sizes:
            try:
                # Stratified sampling
                indices = self._stratified_sample_indices(y_train, n_samples)
                X_subset = X_train.iloc[indices]
                y_subset = y_train.iloc[indices]

                train_data = lgb.Dataset(
                    X_subset,
                    label=y_subset,
                    categorical_feature=self.config.categorical_features,
                    free_raw_data=True
                )

                # Adjust early stopping based on sample size
                early_stopping = max(
                    min_early_stopping,
                    int(self.config.LGBM.early_stopping * (n_samples / len(X_train)))
                )

                subset_model = lgb.train(
                    model_params,
                    train_data,
                    num_boost_round=self.config.LGBM.num_boost_round,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    callbacks=[
                        lgb.early_stopping(early_stopping),
                        lgb.log_evaluation(self.config.LGBM.log_evaluation)
                    ]
                )

                train_pred = subset_model.predict(X_subset)
                val_pred = subset_model.predict(X_val)

                train_score = self._calculate_model_score(y_subset, train_pred)
                val_score = self._calculate_model_score(y_val, val_pred)

                results.add_learning_curve_point(
                    train_size=n_samples,
                    train_score=train_score,
                    val_score=val_score,
                    fold=fold
                )

                del subset_model, train_data

            except Exception as e:
                structured_log(logger, logging.ERROR,
                            f"Error in LightGBM learning curve at size {n_samples}",
                            error=str(e))
                raise

        return results

    def _generate_sklearn_learning_curve_data(self, X_train, y_train, X_val, y_val, fold : int, model_params,
                                            results) -> ModelTrainingResults:
        """Generate learning curve data for sklearn models with consistent handling."""

        structured_log(logger, logging.INFO, 
                    "Starting sklearn learning curve generation",
                    fold=fold)

        # Calculate absolute sizes instead of percentages
        min_samples = max(int(len(X_train) * self.config.learning_curve.min_size), 
                        self.config.learning_curve.min_absolute_samples)

        max_samples = min(int(len(X_train) * self.config.learning_curve.max_size),
                        len(X_train))
        train_sizes = np.linspace(min_samples, max_samples,
                                self.config.learning_curve.n_points, dtype=int)

        # Set random seed for reproducibility
        np.random.seed(self.config.random_state + fold)
        
        # Get the appropriate model class once
        if hasattr(tree, results.model_name):
            model_class = getattr(tree, results.model_name)
        elif hasattr(ensemble, results.model_name):
            model_class = getattr(ensemble, results.model_name)
        else:
            raise ValueError(f"Unsupported model: {results.model_name}")

        for n_samples in train_sizes:
            try:
                # Stratified sampling
                indices = self._stratified_sample_indices(y_train, n_samples)
                X_subset = X_train.iloc[indices]
                y_subset = y_train.iloc[indices]

                model = model_class(**model_params)
                model.fit(X_subset, y_subset)

                # Consistent prediction handling
                train_pred = (model.predict_proba(X_subset)[:, 1] 
                            if hasattr(model, 'predict_proba') 
                            else model.predict(X_subset))
                val_pred = (model.predict_proba(X_val)[:, 1]
                        if hasattr(model, 'predict_proba')
                        else model.predict(X_val))

                train_score = self._calculate_model_score(y_subset, train_pred)
                val_score = self._calculate_model_score(y_val, val_pred)

                results.add_learning_curve_point(
                    train_size=n_samples,
                    train_score=train_score,
                    val_score=val_score,
                    fold=fold
                )

                del model

            except Exception as e:
                structured_log(logger, logging.ERROR,
                            f"Error in sklearn learning curve at size {n_samples}",
                            error=str(e))
                raise

        return results

    def _stratified_sample_indices(self, y: pd.Series, n_samples: int) -> np.ndarray:
        """Generate stratified sample indices maintaining class distribution."""
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # If n_samples equals the total samples, return all indices
        if n_samples >= len(y):
            return np.arange(len(y))
        
        # Otherwise use StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=None)
        indices, _ = next(sss.split(np.zeros(len(y)), y))
        return indices

    def _calculate_model_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate consistent model score across all implementations."""
        return accuracy_score(y_true, (y_pred >= 0.5).astype(int))