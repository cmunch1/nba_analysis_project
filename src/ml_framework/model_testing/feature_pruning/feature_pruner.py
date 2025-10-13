"""Concrete implementation of feature pruning functionality."""

import logging
import traceback
from typing import List, Tuple
import pandas as pd
import numpy as np

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

from .base_feature_pruner import BaseFeaturePruner


class FeaturePruner(BaseFeaturePruner):
    """
    Concrete implementation of feature pruning.

    Identifies low-quality features from audit and removes them from datasets.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize feature pruner.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler factory
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

        # Get pruning configuration
        self.pruning_config = config.core.model_testing_config.feature_pruning

        self.app_logger.structured_log(
            logging.INFO,
            "FeaturePruner initialized",
            config_type=type(config).__name__,
            threshold=self.pruning_config.drop_candidate_threshold
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging."""
        def wrapper(*args, **kwargs):
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    def identify_drop_candidates(self,
                                 audit_df: pd.DataFrame,
                                 threshold: int = None) -> List[str]:
        """
        Identify features to drop based on audit scores.

        Automatically protects critical columns like sort_columns, target, and primary_id.

        Args:
            audit_df: Feature audit DataFrame
            threshold: Minimum drop_candidate_score (uses config default if None)

        Returns:
            List of feature names to drop
        """
        try:
            if threshold is None:
                threshold = self.pruning_config.drop_candidate_threshold

            # Filter features meeting drop threshold
            drop_candidates = audit_df[
                audit_df['drop_candidate_score'] >= threshold
            ]['feature_name'].tolist()

            # Protect critical columns (sort columns, target prefix columns, etc.)
            protected_columns = self._get_protected_columns()

            # Filter out protected columns
            drop_candidates_filtered = [f for f in drop_candidates if f not in protected_columns]
            n_protected = len(drop_candidates) - len(drop_candidates_filtered)

            if n_protected > 0:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Protected columns excluded from drop candidates",
                    n_protected=n_protected,
                    protected_columns=protected_columns
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Identified drop candidates",
                threshold=threshold,
                n_candidates=len(drop_candidates_filtered),
                n_protected=n_protected,
                pct_of_features=f"{len(drop_candidates_filtered)/len(audit_df)*100:.1f}%"
            )

            return drop_candidates_filtered

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'feature_selection',
                "Failed to identify drop candidates",
                error_message=str(e),
                traceback=traceback.format_exc()
            )

    def _get_protected_columns(self) -> set:
        """
        Get set of columns that should never be dropped.

        Returns:
            Set of protected column names
        """
        protected = set()
        model_cfg = self.config.core.model_testing_config

        # Protect sort columns (needed for TimeSeriesSplit)
        if hasattr(model_cfg, 'sort_columns') and model_cfg.sort_columns:
            protected.update(model_cfg.sort_columns)

        # Protect primary ID column
        if hasattr(model_cfg, 'primary_id_column') and model_cfg.primary_id_column:
            protected.add(model_cfg.primary_id_column)

        # Protect target column (though it should already be excluded from features)
        if hasattr(self.config, 'target_column') and self.config.target_column:
            protected.add(self.config.target_column)
            # Also protect with home prefix
            if hasattr(self.config, 'home_team_prefix'):
                protected.add(f"{self.config.home_team_prefix}{self.config.target_column}")

        return protected

    def prune_dataset(self,
                     df: pd.DataFrame,
                     features_to_drop: List[str]) -> pd.DataFrame:
        """
        Remove specified features from dataset.

        Args:
            df: Input DataFrame
            features_to_drop: List of feature names to remove

        Returns:
            DataFrame with features removed
        """
        try:
            # Only drop features that exist in the DataFrame
            features_in_df = [f for f in features_to_drop if f in df.columns]
            features_missing = [f for f in features_to_drop if f not in df.columns]

            if features_missing:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Some features to drop not found in dataset",
                    n_missing=len(features_missing),
                    examples=features_missing[:5]
                )

            pruned_df = df.drop(columns=features_in_df)

            self.app_logger.structured_log(
                logging.INFO,
                "Dataset pruned",
                original_shape=df.shape,
                pruned_shape=pruned_df.shape,
                n_features_dropped=len(features_in_df)
            )

            return pruned_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to prune dataset",
                error_message=str(e),
                traceback=traceback.format_exc(),
                n_features_to_drop=len(features_to_drop)
            )

    def validate_pruning_safety(self,
                                original_features: List[str],
                                features_to_drop: List[str]) -> Tuple[bool, str]:
        """
        Validate that pruning is safe (not too aggressive).

        Args:
            original_features: Original feature list
            features_to_drop: Features proposed for dropping

        Returns:
            Tuple of (is_safe: bool, message: str)
        """
        try:
            n_original = len(original_features)
            n_to_drop = len(features_to_drop)
            n_remaining = n_original - n_to_drop
            pct_to_drop = n_to_drop / n_original

            # Check minimum features constraint
            min_features = self.pruning_config.min_features_to_keep
            if n_remaining < min_features:
                msg = f"Would leave only {n_remaining} features (minimum: {min_features})"
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Pruning safety check failed: too few features remaining",
                    n_remaining=n_remaining,
                    min_required=min_features
                )
                return False, msg

            # Check maximum drop percentage constraint
            max_drop_pct = self.pruning_config.max_features_to_drop_pct
            if pct_to_drop > max_drop_pct:
                msg = f"Would drop {pct_to_drop:.1%} of features (maximum: {max_drop_pct:.1%})"
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Pruning safety check failed: too many features to drop",
                    pct_to_drop=f"{pct_to_drop:.1%}",
                    max_allowed=f"{max_drop_pct:.1%}"
                )
                return False, msg

            # Passed all checks
            self.app_logger.structured_log(
                logging.INFO,
                "Pruning safety check passed",
                n_original=n_original,
                n_to_drop=n_to_drop,
                n_remaining=n_remaining,
                pct_to_drop=f"{pct_to_drop:.1%}"
            )
            return True, "Pruning is safe"

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'feature_selection',
                "Failed to validate pruning safety",
                error_message=str(e),
                traceback=traceback.format_exc()
            )

    def get_pruning_summary(self,
                           original_features: List[str],
                           features_to_drop: List[str],
                           audit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary of features being dropped.

        Args:
            original_features: Original feature list
            features_to_drop: Features to drop
            audit_df: Feature audit DataFrame

        Returns:
            Summary DataFrame with drop reasons
        """
        try:
            # Filter audit for features being dropped
            drop_summary = audit_df[audit_df['feature_name'].isin(features_to_drop)].copy()

            # Sort by drop_candidate_score (highest first)
            drop_summary = drop_summary.sort_values('drop_candidate_score', ascending=False)

            # Select relevant columns
            summary_columns = [
                'feature_name',
                'drop_candidate_score',
                'near_zero_importance',
                'high_missing_flag',
                'high_collinearity_flag',
                'unstable_flag',
                'leakage_flag',
                'missing_rate',
                'shap_mean_abs',
                'stability_score'
            ]

            # Only include columns that exist
            summary_columns = [col for col in summary_columns if col in drop_summary.columns]
            drop_summary = drop_summary[summary_columns]

            self.app_logger.structured_log(
                logging.INFO,
                "Generated pruning summary",
                n_features_to_drop=len(drop_summary)
            )

            return drop_summary

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to generate pruning summary",
                error_message=str(e),
                traceback=traceback.format_exc()
            )
