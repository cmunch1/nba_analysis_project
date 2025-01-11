import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional, Tuple
import logging
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import PreprocessingError
from .data_classes import PreprocessingResults

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
                        num_pipeline, X[numerical_features], y, preproc_results
                    )
            
            # Add categorical pipeline if there are categorical features
            if len(categorical_features) > 0:
                if cat_pipeline := self._create_categorical_pipeline(model_name):
                    transformers.append(('categorical', cat_pipeline, categorical_features))
                    
                    self._track_categorical_preprocessing(
                        cat_pipeline, X[categorical_features], preproc_results
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
            
            # Apply feature selection if configured
            if hasattr(model_config, 'feature_selection') and model_config.feature_selection:
                transformed = self._apply_feature_selection(
                    transformed, y, model_config.feature_selection, preprocessing_results
                )
            
            # Apply feature engineering if configured
            if hasattr(model_config, 'feature_engineering') and model_config.feature_engineering:
                transformed = self._apply_feature_engineering(
                    transformed, model_config.feature_engineering, preprocessing_results
                )
            
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
                
        if hasattr(model_config.numerical, 'binning'):
            if model_config.numerical.binning:  # Only add if not None/empty
                steps.append(('binner', BinningTransformer(
                    method=model_config.numerical.binning,
                    n_bins=model_config.numerical.n_bins
                )))
            
        return Pipeline(steps) if steps else None
        
    def _create_categorical_pipeline(self, model_name: str) -> Pipeline:
        """Create categorical preprocessing pipeline based on config."""
        model_config = self._get_model_config(model_name)
        steps = []
        
        if hasattr(model_config.categorical, 'handling_missing'):
            if model_config.categorical.handling_missing:  # Only add if not None/empty
                steps.append(('imputer', SimpleImputer(
                    strategy=model_config.categorical.handling_missing,
                    fill_value='missing'
                )))
            
        if hasattr(model_config.categorical, 'encoding'):
            if model_config.categorical.encoding:  # Only add if not None/empty
                if model_config.categorical.encoding == "onehot":
                    max_categories = (model_config.categorical.max_categories 
                                    if hasattr(model_config.categorical, 'max_categories') 
                                    else None)
                    steps.append(('encoder', OneHotEncoder(
                        handle_unknown='ignore',
                        max_categories=max_categories
                    )))
                elif model_config.categorical.encoding == "label":
                    steps.append(('encoder', LabelEncoder()))
                elif model_config.categorical.encoding == "target":
                    steps.append(('encoder', TargetEncoder()))
                elif model_config.categorical.encoding == "frequency":
                    min_frequency = (model_config.categorical.min_frequency 
                                   if hasattr(model_config.categorical, 'min_frequency') 
                                   else None)
                    steps.append(('encoder', FrequencyEncoder(
                        min_frequency=min_frequency
                    )))
            
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
                

    @log_performance
    def _apply_feature_selection(self, X: np.ndarray, y: np.ndarray,
                            method: str, results: Optional['ModelTrainingResults'] = None) -> np.ndarray:
        """
        Apply feature selection based on specified method.
        
        Args:
            X: Input feature array
            y: Target array
            method: Feature selection method ('variance', 'mutual_info', or others)
            results: Optional ModelTrainingResults for tracking
            
        Returns:
            Array with selected features
        """
        structured_log(logger, logging.INFO, "Applying feature selection",
                    method=method,
                    input_shape=X.shape)
        
        try:
            if method == "variance":
                selector = VarianceThreshold(threshold=0.01)  # You might want to make threshold configurable
                transformed = selector.fit_transform(X, y)
                
                if results:
                    results.add_preprocessing_step(
                        name='variance_threshold',
                        step_type='feature_selection',
                        columns=list(range(X.shape[1])),
                        parameters={'threshold': 0.01},
                        statistics={
                            'variances': selector.variances_.tolist(),
                            'selected_features': selector.get_support().tolist()
                        }
                    )
                    
            elif method == "mutual_info":
                # Select top k features based on mutual information
                k = min(50, X.shape[1])  # You might want to make k configurable
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
                transformed = selector.fit_transform(X, y)
                
                if results:
                    results.add_preprocessing_step(
                        name='mutual_info',
                        step_type='feature_selection',
                        columns=list(range(X.shape[1])),
                        parameters={'k': k},
                        statistics={
                            'scores': selector.scores_.tolist(),
                            'selected_features': selector.get_support().tolist()
                        }
                    )
                    
            else:
                raise ValueError(f"Unsupported feature selection method: {method}")
            
            structured_log(logger, logging.INFO, "Feature selection completed",
                        original_features=X.shape[1],
                        selected_features=transformed.shape[1])
            
            return transformed
            
        except Exception as e:
            raise PreprocessingError("Error in feature selection",
                                error_message=str(e),
                                method=method,
                                input_shape=X.shape)

    @log_performance
    def _apply_feature_engineering(self, X: np.ndarray,
                                methods: List[str],
                                results: Optional['ModelTrainingResults'] = None) -> np.ndarray:
        """
        Apply feature engineering methods to the data.
        
        Args:
            X: Input feature array
            methods: List of feature engineering methods to apply
            results: Optional ModelTrainingResults for tracking
            
        Returns:
            Array with engineered features added
        """
        structured_log(logger, logging.INFO, "Starting feature engineering",
                    methods=methods,
                    input_shape=X.shape)
        
        try:
            transformed = X
            
            for method in methods:
                if method == "interactions":
                    transformed = self._create_interactions(transformed)
                    if results:
                        results.add_preprocessing_step(
                            name='interactions',
                            step_type='feature_engineering',
                            columns=list(range(X.shape[1])),
                            parameters={'method': 'interactions'},
                            statistics={
                                'n_original_features': X.shape[1],
                                'n_interaction_features': transformed.shape[1] - X.shape[1]
                            }
                        )
                        
                elif method == "polynomials":
                    transformed = self._create_polynomials(transformed)
                    if results:
                        results.add_preprocessing_step(
                            name='polynomials',
                            step_type='feature_engineering',
                            columns=list(range(X.shape[1])),
                            parameters={'method': 'polynomials', 'degree': 2},
                            statistics={
                                'n_original_features': X.shape[1],
                                'n_polynomial_features': transformed.shape[1] - X.shape[1]
                            }
                        )
                else:
                    raise ValueError(f"Unsupported feature engineering method: {method}")
            
            structured_log(logger, logging.INFO, "Feature engineering completed",
                        original_shape=X.shape,
                        final_shape=transformed.shape)
            
            return transformed
            
        except Exception as e:
            raise PreprocessingError("Error in feature engineering",
                                error_message=str(e),
                                methods=methods,
                                input_shape=X.shape)
    
    def _create_interactions(self, X: np.ndarray) -> np.ndarray:
        """Create interaction terms between features"""
        from itertools import combinations
        n_features = X.shape[1]
        interactions = []
        for i, j in combinations(range(n_features), 2):
            interactions.append(X[:, i] * X[:, j])
        return np.column_stack([X] + interactions)
    
    def _create_polynomials(self, X: np.ndarray, degree: int = 2, 
                        interaction_only: bool = False, 
                        include_bias: bool = False) -> np.ndarray:
        """
        Create polynomial features from the input data.
        
        Args:
            X: Input feature array of shape (n_samples, n_features)
            degree: Maximum degree of polynomial features (default: 2)
            interaction_only: If True, only interaction features are produced, 
                            no individual features in higher degrees (default: False)
            include_bias: If True, include a bias (constant) column (default: False)
            
        Returns:
            Array containing original and polynomial features
            
        Example:
            For input features [a, b] with degree=2, generates:
            [a, b, a², ab, b²] (or [a, b, ab] if interaction_only=True)
        """
        structured_log(logger, logging.INFO, "Creating polynomial features",
                    input_shape=X.shape,
                    degree=degree,
                    interaction_only=interaction_only)
        
        try:
            # Input validation
            if not isinstance(X, np.ndarray):
                X = np.asarray(X)
            
            if X.ndim != 2:
                raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
            
            if degree < 1:
                raise ValueError(f"Degree must be >= 1, got {degree}")
                
            n_samples, n_features = X.shape
            
            # Early return if degree is 1 and no bias term is needed
            if degree == 1 and not include_bias:
                return X
                
            # Calculate expected number of features for logging
            if interaction_only:
                from scipy.special import comb
                n_output_features = sum(comb(n_features, i) for i in range(1, degree + 1))
            else:
                from itertools import combinations_with_replacement
                n_output_features = sum(comb(n_features + i - 1, i) 
                                    for i in range(1, degree + 1))
            
            if include_bias:
                n_output_features += 1
                
            # Check if output size is reasonable
            output_size_bytes = n_output_features * n_samples * X.dtype.itemsize
            if output_size_bytes > 1e9:  # 1GB warning threshold
                structured_log(logger, logging.WARNING, 
                            "Large polynomial feature matrix expected",
                            expected_size_gb=output_size_bytes/1e9,
                            n_output_features=n_output_features)
                
            # Create polynomial features
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=include_bias
            )
            
            transformed = poly.fit_transform(X)
            
            # Log transformation results
            structured_log(logger, logging.INFO, 
                        "Polynomial features created",
                        input_shape=X.shape,
                        output_shape=transformed.shape,
                        n_new_features=transformed.shape[1] - X.shape[1])
            
            # Store feature names if available
            if hasattr(self, '_current_results') and self._current_results is not None:
                if hasattr(poly, 'get_feature_names_out'):
                    feature_names = poly.get_feature_names_out()
                else:
                    feature_names = poly.get_feature_names()
                    
                self._current_results.preprocessing_info.engineered_features.update({
                    'polynomials': feature_names.tolist()
                })
            
            return transformed
            
        except Exception as e:
            raise PreprocessingError("Error creating polynomial features",
                                error_message=str(e),
                                input_shape=X.shape,
                                degree=degree,
                                interaction_only=interaction_only)

    def _estimate_poly_feature_count(self, n_features: int, degree: int, 
                                interaction_only: bool = False) -> int:
        """
        Estimate the number of polynomial features that will be generated.
        
        Args:
            n_features: Number of input features
            degree: Maximum polynomial degree
            interaction_only: If True, only interaction terms are counted
            
        Returns:
            Expected number of features after polynomial transformation
        """
        if interaction_only:
            from scipy.special import comb
            return int(sum(comb(n_features, i) for i in range(1, degree + 1)))
        else:
            return int(sum(comb(n_features + i - 1, i) for i in range(1, degree + 1)))    

    def _track_numerical_preprocessing(self, pipeline: Pipeline, X: pd.DataFrame, 
                                     results: PreprocessingResults) -> None:
        """Track numerical preprocessing steps."""
        for name, transformer in pipeline.steps:
            statistics = {}
            if hasattr(transformer, 'mean_'):
                statistics['mean'] = transformer.mean_.tolist()
            if hasattr(transformer, 'scale_'):
                statistics['scale'] = transformer.scale_.tolist()
                
            step = PreprocessingStep(
                name=name,
                type='numerical',
                columns=X.columns.tolist(),
                parameters=transformer.get_params(),
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
                
            step = PreprocessingStep(
                name=name,
                type='categorical',
                columns=X.columns.tolist(),
                parameters=transformer.get_params(),
                statistics=statistics
            )
            results.steps.append(step)