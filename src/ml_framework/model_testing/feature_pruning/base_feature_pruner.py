"""Abstract base class for feature pruning."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class BaseFeaturePruner(ABC):
    """
    Abstract base class for feature pruning functionality.

    Identifies and removes low-quality features based on feature audit results.
    """

    @abstractmethod
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize feature pruner with core dependencies.

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
    def identify_drop_candidates(self,
                                 audit_df: pd.DataFrame,
                                 threshold: int = None) -> List[str]:
        """
        Identify features to drop based on audit scores.

        Args:
            audit_df: Feature audit DataFrame
            threshold: Minimum drop_candidate_score (uses config default if None)

        Returns:
            List of feature names to drop
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
