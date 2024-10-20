import numpy as np
import logging
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import PreprocessingError

logger = logging.getLogger(__name__)

class Preprocessor:
    @log_performance
    def __init__(self):
        """
        Initialize the Preprocessor class.
        """
        self.preprocessor = None
        structured_log(logger, logging.INFO, "Preprocessor initialized")

    @log_performance
    def _fit(self, X: np.ndarray, numerical_features: List[str], categorical_features: List[str]) -> None:
        """
        Fit the preprocessor to the data.
        
        Args:
            X (np.ndarray): The input features
            numerical_features (List[str]): List of column names for numerical features
            categorical_features (List[str]): List of column names for categorical features
        """
        structured_log(logger, logging.INFO, "Starting preprocessor fitting",
                       input_shape=X.shape)
        try:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            self.preprocessor.fit(X)
            structured_log(logger, logging.INFO, "Preprocessor fitting completed")
        except Exception as e:
            raise PreprocessingError("Error in preprocessor fitting",
                                     error_message=str(e),
                                     input_shape=X.shape)

    @log_performance
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            X (np.ndarray): The input features to transform

        Returns:
            np.ndarray: The transformed features
        """
        structured_log(logger, logging.INFO, "Starting data transformation",
                       input_shape=X.shape)
        try:
            transformed_X = self.preprocessor.transform(X)
            structured_log(logger, logging.INFO, "Data transformation completed",
                           output_shape=transformed_X.shape)
            return transformed_X
        except Exception as e:
            raise PreprocessingError("Error in data transformation",
                                     error_message=str(e),
                                     input_shape=X.shape)

    @log_performance
    def fit_transform(self, X: np.ndarray, numerical_features: List[str], categorical_features: List[str]) -> np.ndarray:
        """
        Fit the preprocessor to the data and then transform it.
        
        Args:
            X (np.ndarray): The input features
            numerical_features (List[str]): List of column names for numerical features
            categorical_features (List[str]): List of column names for categorical features

        Returns:
            np.ndarray: The transformed features
        """
        structured_log(logger, logging.INFO, "Starting fit_transform",
                       input_shape=X.shape)
        try:
            self._fit(X, numerical_features, categorical_features)
            return self._transform(X)
        except Exception as e:
            raise PreprocessingError("Error in fit_transform",
                                     error_message=str(e),
                                     input_shape=X.shape)
