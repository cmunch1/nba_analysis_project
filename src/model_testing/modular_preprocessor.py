"""
This module contains the ModularPreprocessor class, which is used to preprocess data using a modular approach.
It allows for the creation of preprocessing pipelines for different models and the tracking of preprocessing steps.

I have decided to not use a lot of the functionality of the preprocessor as originally conceived.

I have made the philosophical decision to put all functionality that changes the number of columns in the data 
in the feature engineering module.

The focus here will be simple transformations that do not change the number of columns that can be easily
implemented at "runtime" - this is to allow for easy iteration through a lot of tests without having to
do any work on the data files in-between. E.g. I can script a series of runs that test with StandardScaler, 
then run a test with MinMaxScaler, then run a test with RobustScaler, etc.

Also, in production, I can use a binary object to store the preprocessor and load it in when needed.

"""








import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional, Tuple
import logging
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import PreprocessingError
from .data_classes import PreprocessingResults, PreprocessingStep

logger = logging.getLogger(__name__)

class ModularPreprocessor:
    @log_performance
    def __init__(self, config):
        """
        Initialize with Config instance.
        
        Args:
            config: Config instance containing preprocessing settings
        """
        self.config = config
        self.fitted_transformers = {}
        self.preprocessor = None
        self._current_results = None
        structured_log(logger, logging.INFO, "ModularPreprocessor initialized")

    @log_performance
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                    model_name: str = None, preprocessing_results: PreprocessingResults = None) -> Tuple[pd.DataFrame, PreprocessingResults]:
        """
        Fit and transform the data according to the preprocessing config for the specified model.
        
        Args:
            X: Input features DataFrame
            y: Optional target Series for supervised preprocessing steps
            model_name: Name of the model to get specific preprocessing config
            
        Returns:
            Transformed DataFrame with preprocessed features
        """
        structured_log(logger, logging.INFO, "Starting preprocessing",
                      input_shape=X.shape,
                      model_name=model_name)
        
        try:
            
            if preprocessing_results is None:
                preprocessing_results = PreprocessingResults()
            preprocessing_results.original_features = X.columns.tolist()
            
            # Split features
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object', 'category']).columns

            print('categorical_features', categorical_features)
            
            # Create transformers list
            transformers = []
            
            # Add numerical pipeline if there are numerical features
            if len(numerical_features) > 0:
                if num_pipeline := self._create_numerical_pipeline(model_name):
                    transformers.append(('numerical', num_pipeline, numerical_features))
                    
                    self._track_numerical_preprocessing(
                        num_pipeline, X[numerical_features], y, preprocessing_results
                    )
            
            # Add categorical pipeline if there are categorical features
            if len(categorical_features) > 0:
                if cat_pipeline := self._create_categorical_pipeline(model_name):
                    transformers.append(('categorical', cat_pipeline, categorical_features))
                    
                    self._track_categorical_preprocessing(
                        cat_pipeline, X[categorical_features], preprocessing_results
                    )
            
            # Create and fit ColumnTransformer
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'  # Keep any columns not explicitly transformed
            )
            
            # Fit and transform the data
            transformed = self.preprocessor.fit_transform(X, y)
            
            # Get model-specific config for additional preprocessing steps
            model_config = self._get_model_config(model_name)
                   
            # Generate feature names for the transformed data
            feature_names = self._get_feature_names(X, self.preprocessor)
            
            # Update results with final feature names if tracking
            preprocessing_results.final_features = feature_names
            
            # Convert to DataFrame with proper feature names
            transformed_df = pd.DataFrame(transformed, columns=feature_names, index=X.index)
            
            structured_log(logger, logging.INFO, "Preprocessing completed",
                        output_shape=transformed_df.shape,
                        n_features=len(feature_names))
            
            preprocessing_results.final_shape = transformed_df.shape
            
            structured_log(logger, logging.INFO, "Preprocessing results",
                          results=preprocessing_results.to_dict())
            
            return transformed_df, preprocessing_results
            
        except Exception as e:
            raise PreprocessingError("Error in preprocessing pipeline",
                                   error_message=str(e),
                                   input_shape=X.shape)


    @log_performance
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed DataFrame with preprocessed features
        """
        structured_log(logger, logging.INFO, "Starting transform of new data",
                    input_shape=X.shape)
        
        try:
            if not hasattr(self, 'preprocessor'):
                raise PreprocessingError("Preprocessor has not been fitted. Call fit_transform first.")
            
            # Transform the data using the fitted preprocessor
            transformed = self.preprocessor.transform(X)
            
            # Get feature names
            feature_names = self._get_feature_names(X, self.preprocessor)
            
            # Convert to DataFrame with proper feature names
            transformed_df = pd.DataFrame(transformed, columns=feature_names, index=X.index)
            
            structured_log(logger, logging.INFO, "Transform completed",
                        output_shape=transformed_df.shape)
            
            return transformed_df
            
        except Exception as e:
            raise PreprocessingError("Error in transform pipeline",
                                error_message=str(e),
                                input_shape=X.shape)

        
    def _get_model_config(self, model_name: str):
        """Get preprocessing config for specific model, falling back to default if not specified."""
        if (hasattr(self.config.preprocessing, 'model_specific') and 
            hasattr(self.config.preprocessing.model_specific, model_name)):
            return getattr(self.config.preprocessing.model_specific, model_name)
        return self.config.preprocessing.default
        
    def _create_numerical_pipeline(self, model_name: str) -> Pipeline:
        """Create numerical preprocessing pipeline based on config."""
        model_config = self._get_model_config(model_name)
        steps = []
        
        if hasattr(model_config.numerical, 'handling_missing'):
            if model_config.numerical.handling_missing:  # Only add if not None/empty
                steps.append(('imputer', SimpleImputer(
                    strategy=model_config.numerical.handling_missing
                )))
            
        if hasattr(model_config.numerical, 'handling_outliers'):
            if model_config.numerical.handling_outliers:  # Only add if not None/empty
                if model_config.numerical.handling_outliers == "winsorize":
                    steps.append(('outliers', WinsorizationTransformer()))
                elif model_config.numerical.handling_outliers == "clip":
                    steps.append(('outliers', ClippingTransformer()))
                
        if hasattr(model_config.numerical, 'scaling'):
            if model_config.numerical.scaling:  # Only add if not None/empty
                if model_config.numerical.scaling == "standard":
                    steps.append(('scaler', StandardScaler()))
                elif model_config.numerical.scaling == "minmax":
                    steps.append(('scaler', MinMaxScaler()))
                elif model_config.numerical.scaling == "robust":
                    steps.append(('scaler', RobustScaler()))
                elif model_config.numerical.scaling == "maxabs":
                    steps.append(('scaler', MaxAbsScaler()))
                
            
        return Pipeline(steps) if steps else None
        
    def _create_categorical_pipeline(self, model_name: str) -> Pipeline:
        """Create categorical preprocessing pipeline based on config."""
        model_config = self._get_model_config(model_name)
        steps = []
        
        if hasattr(model_config.categorical, 'handling_missing'):
            if model_config.categorical.handling_missing == "constant":
                steps.append(('imputer', SimpleImputer(
                    strategy='constant',
                    fill_value='missing'
                )))
            
        if hasattr(model_config.categorical, 'encoding'):
            if model_config.categorical.encoding:
                if model_config.categorical.encoding == "ordinal":
                    steps.append(('encoder', OrdinalEncoder(
                        handle_unknown='use_encoded_value', 
                        unknown_value=-1
                    )))
                elif model_config.categorical.encoding == "target":
                    steps.append(('encoder', TargetEncoder()))
                elif model_config.categorical.encoding == "frequency":
                    min_frequency = (model_config.categorical.min_frequency 
                                   if hasattr(model_config.categorical, 'min_frequency') 
                                   else None)
                    steps.append(('encoder', FrequencyEncoder(
                        min_frequency=min_frequency
                    )))
                elif model_config.categorical.encoding == "label":
                    steps.append(('encoder', LabelEncoder()))
        
        return Pipeline(steps) if steps else None
    

    def _get_feature_names(self, X: pd.DataFrame, column_transformer: ColumnTransformer) -> List[str]:
        """
        Get feature names after preprocessing transformations.
        
        Args:
            X: Original input DataFrame
            column_transformer: Fitted ColumnTransformer
            
        Returns:
            List of feature names after transformations
        """
        feature_names = []
    
        # Process each transformer in the ColumnTransformer
        for name, transformer, features in column_transformer.transformers_:
            if name == 'remainder':
                if features:
                    feature_names.extend(features)
                continue
                
            # Get the feature names based on transformer type
            if hasattr(transformer, 'get_feature_names_out'):
                # For newer scikit-learn versions
                trans_features = transformer.get_feature_names_out()
            elif hasattr(transformer, 'get_feature_names'):
                # For older scikit-learn versions
                trans_features = transformer.get_feature_names()
            else:
                # Fallback: use numbered features
                trans_features = [f"{name}{i}" for i in range(transformer.n_features_out_)]
            
            feature_names.extend(trans_features)
        
        return feature_names        
                
 
    def _track_numerical_preprocessing(self, pipeline: Pipeline, X: pd.DataFrame, 
                                     results: PreprocessingResults) -> None:
        """Track numerical preprocessing steps."""
        for name, transformer in pipeline.steps:
            statistics = {}
            if hasattr(transformer, 'mean_'):
                statistics['mean'] = transformer.mean_.tolist()
            if hasattr(transformer, 'scale_'):
                statistics['scale'] = transformer.scale_.tolist()
                
            # Convert parameters to JSON-serializable format
            params = {}
            for key, value in transformer.get_params().items():
                if isinstance(value, type):
                    params[key] = str(value)
                else:
                    params[key] = value
                    
            step = PreprocessingStep(
                name=name,
                type='numerical',
                columns=X.columns.tolist(),
                parameters=params,  # Use the converted parameters
                statistics=statistics
            )
            results.steps.append(step)
    
    def _track_categorical_preprocessing(self, pipeline: Pipeline, X: pd.DataFrame, 
                                       results: PreprocessingResults) -> None:
        """Track categorical preprocessing steps."""
        for name, transformer in pipeline.steps:
            statistics = {}
            if hasattr(transformer, 'categories_'):
                statistics['categories'] = [cat.tolist() for cat in transformer.categories_]
                
            # Convert parameters to JSON-serializable format
            params = {}
            for key, value in transformer.get_params().items():
                if isinstance(value, type):
                    params[key] = str(value)
                else:
                    params[key] = value
                    
            step = PreprocessingStep(
                name=name,
                type='categorical',
                columns=X.columns.tolist(),
                parameters=params,  # Use the converted parameters
                statistics=statistics
            )
            results.steps.append(step)