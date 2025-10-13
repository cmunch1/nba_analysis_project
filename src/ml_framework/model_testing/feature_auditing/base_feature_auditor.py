"""Abstract base class for feature auditing."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.framework.data_classes.training import ModelTrainingResults


class BaseFeatureAuditor(ABC):
    """
    Abstract base class for feature auditing functionality.

    Analyzes features used in model training to identify:
    - Feature importance (SHAP, permutation, model-specific)
    - Statistical properties (coverage, variance, cardinality)
    - Redundancy (correlation, VIF)
    - Stability (across CV folds)
    - Drop candidates (composite scoring)
    """

    @abstractmethod
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize feature auditor with core dependencies.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler factory
        """
        pass

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        pass

    @abstractmethod
    def create_audit(self,
                     results: ModelTrainingResults,
                     X_eval: pd.DataFrame,
                     y_eval: pd.Series,
                     run_id: Optional[str] = None,
                     experiment_id: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive feature audit from training results.

        Args:
            results: Model training results containing model, predictions, SHAP values, etc.
            X_eval: Evaluation dataset (for permutation importance)
            y_eval: Evaluation targets (for permutation importance)
            run_id: MLflow run identifier (optional)
            experiment_id: MLflow experiment identifier (optional)

        Returns:
            DataFrame containing feature audit with columns:
                - feature_name: Feature name
                - data_type: numeric/categorical/binary
                - coverage: Non-null ratio
                - missing_rate: Null ratio
                - unique_values: Number of unique values
                - variance: Variance (numeric only)
                - cardinality: Unique values (categorical)
                - shap_mean_abs: Mean absolute SHAP value
                - shap_std_abs: Std of absolute SHAP values
                - shap_rank: Rank by SHAP importance
                - permutation_importance_mean: Mean permutation importance
                - permutation_importance_std: Std of permutation importance
                - model_gain_importance: Model-specific importance (e.g., XGBoost gain)
                - pairwise_max_corr: Max correlation with any other feature
                - target_correlation: Correlation with target
                - stability_score: Fraction of CV folds where feature is in top-k
                - near_zero_importance: Flag for low importance
                - high_missing_flag: Flag for high missing rate
                - high_collinearity_flag: Flag for high correlation
                - unstable_flag: Flag for low stability
                - drop_candidate_score: Composite score for drop priority
                - model_name: Model type (e.g., XGBoost)
                - run_id: MLflow run ID
                - experiment_id: MLflow experiment ID
                - audit_timestamp: When audit was created
        """
        pass

    @abstractmethod
    def save_audit(self,
                   audit_df: pd.DataFrame,
                   model_name: str,
                   eval_type: str,
                   run_id: Optional[str] = None) -> str:
        """
        Save feature audit to file.

        Args:
            audit_df: Feature audit DataFrame
            model_name: Model name (e.g., XGBoost)
            eval_type: Evaluation type ('oof' or 'validation')
            run_id: MLflow run identifier (optional)

        Returns:
            Path to saved audit file
        """
        pass

    @abstractmethod
    def compute_basic_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic statistical metrics for features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with columns: feature_name, data_type, coverage,
            missing_rate, unique_values, variance, cardinality
        """
        pass

    @abstractmethod
    def compute_importance_metrics(self,
                                   results: ModelTrainingResults,
                                   X_eval: pd.DataFrame,
                                   y_eval: pd.Series) -> pd.DataFrame:
        """
        Compute feature importance metrics.

        Args:
            results: Training results containing SHAP values and model
            X_eval: Evaluation features
            y_eval: Evaluation targets

        Returns:
            DataFrame with columns: feature_name, shap_mean_abs, shap_std_abs,
            shap_rank, permutation_importance_mean, permutation_importance_std,
            model_gain_importance
        """
        pass

    @abstractmethod
    def compute_redundancy_metrics(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature redundancy metrics.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with columns: feature_name, pairwise_max_corr, vif (optional)
        """
        pass

    @abstractmethod
    def compute_stability_scores(self,
                                 results: ModelTrainingResults,
                                 fold_importances: Optional[List[Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Compute stability scores across CV folds.

        Args:
            results: Training results
            fold_importances: List of importance dictionaries per fold (optional)

        Returns:
            DataFrame with columns: feature_name, stability_score
        """
        pass

    @abstractmethod
    def compute_drop_candidate_scores(self, audit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite drop candidate scores and flags.

        Args:
            audit_df: Partial audit DataFrame with computed metrics

        Returns:
            DataFrame with added columns: near_zero_importance, high_missing_flag,
            high_collinearity_flag, unstable_flag, drop_candidate_score
        """
        pass
