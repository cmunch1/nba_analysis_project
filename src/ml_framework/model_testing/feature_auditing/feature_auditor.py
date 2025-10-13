"""Concrete implementation of feature auditing functionality."""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.framework.data_classes.training import ModelTrainingResults

from .base_feature_auditor import BaseFeatureAuditor
from . import audit_metrics


class FeatureAuditor(BaseFeatureAuditor):
    """
    Concrete implementation of feature auditing.

    Analyzes features from model training results and generates comprehensive
    audit reports for feature selection and pruning.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize feature auditor.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler factory
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

        # Get feature audit configuration
        self.audit_config = config.core.model_testing_config.feature_audit_metrics
        self.threshold_config = config.core.model_testing_config.feature_audit_thresholds

        self.app_logger.structured_log(
            logging.INFO,
            "FeatureAuditor initialized",
            config_type=type(config).__name__
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging."""
        def wrapper(*args, **kwargs):
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def create_audit(self,
                     results: ModelTrainingResults,
                     X_eval: pd.DataFrame,
                     y_eval: pd.Series,
                     run_id: Optional[str] = None,
                     experiment_id: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive feature audit from training results.

        Args:
            results: Model training results
            X_eval: Evaluation dataset
            y_eval: Evaluation targets
            run_id: MLflow run identifier
            experiment_id: MLflow experiment identifier

        Returns:
            Feature audit DataFrame
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Creating feature audit",
            model_name=results.model_name,
            n_features=len(results.feature_names),
            has_shap=results.shap_values is not None
        )

        try:
            # Initialize audit with basic stats
            audit_df = self.compute_basic_stats(results.feature_data)

            # Add importance metrics
            importance_df = self.compute_importance_metrics(results, X_eval, y_eval)
            audit_df = audit_df.merge(importance_df, on='feature_name', how='left')

            # Add redundancy metrics
            if self.audit_config.compute_correlation_matrix:
                redundancy_df = self.compute_redundancy_metrics(results.feature_data)
                audit_df = audit_df.merge(redundancy_df, on='feature_name', how='left')

            # Add target correlation
            if self.audit_config.compute_target_correlation:
                target_corr = audit_metrics.compute_target_correlation(
                    results.feature_data,
                    results.target_data,
                    method=self.audit_config.correlation_method
                )
                audit_df['target_correlation'] = audit_df['feature_name'].map(target_corr)

            # Add stability scores (if OOF CV was performed)
            if self.audit_config.compute_stability_score and results.n_folds > 0:
                stability_df = self.compute_stability_scores(results)
                audit_df = audit_df.merge(stability_df, on='feature_name', how='left')
            else:
                audit_df['stability_score'] = np.nan

            # Compute drop candidate scores
            audit_df = self.compute_drop_candidate_scores(audit_df)

            # Add metadata
            audit_df['model_name'] = results.model_name
            audit_df['run_id'] = run_id or ''
            audit_df['experiment_id'] = experiment_id or ''
            audit_df['audit_timestamp'] = datetime.now().isoformat()

            # Sort by SHAP importance
            if 'shap_mean_abs' in audit_df.columns:
                audit_df = audit_df.sort_values('shap_mean_abs', ascending=False)

            self.app_logger.structured_log(
                logging.INFO,
                "Feature audit created successfully",
                n_features=len(audit_df),
                n_drop_candidates=int((audit_df['drop_candidate_score'] >=
                                      self.threshold_config.drop_candidate_threshold).sum())
            )

            return audit_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Feature audit creation failed",
                error_message=str(e),
                traceback=traceback.format_exc(),
                model_name=results.model_name
            )

    def save_audit(self,
                   audit_df: pd.DataFrame,
                   model_name: str,
                   eval_type: str,
                   run_id: Optional[str] = None) -> str:
        """
        Save feature audit to file.

        Args:
            audit_df: Feature audit DataFrame
            model_name: Model name
            eval_type: Evaluation type ('oof' or 'validation')
            run_id: MLflow run identifier

        Returns:
            Path to saved audit file
        """
        try:
            # Create output directory
            output_dir = Path(self.config.core.model_testing_config.feature_audit_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_parts = [
                'feature_audit',
                model_name,
                eval_type
            ]

            if self.config.core.model_testing_config.feature_audit_versioning and run_id:
                filename_parts.append(run_id[:8])  # Short run ID

            filename_parts.append(timestamp)

            # Determine file format
            file_format = self.config.core.model_testing_config.feature_audit_format
            filename = '_'.join(filename_parts) + f'.{file_format}'
            file_path = output_dir / filename

            # Save based on format
            if file_format == 'parquet':
                audit_df.to_parquet(file_path, index=False)
            else:  # csv
                audit_df.to_csv(file_path, index=False)

            self.app_logger.structured_log(
                logging.INFO,
                "Feature audit saved",
                file_path=str(file_path),
                file_format=file_format,
                n_features=len(audit_df)
            )

            return str(file_path)

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                "Failed to save feature audit",
                error_message=str(e),
                traceback=traceback.format_exc(),
                model_name=model_name
            )

    def compute_basic_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute basic statistical metrics."""
        try:
            # Get categorical features from config
            categorical_features = self.config.core.model_testing_config.categorical_features or []

            # Infer feature types
            feature_types = audit_metrics.infer_feature_types(X, categorical_features)

            # Compute coverage and missing
            coverage, missing_rate = audit_metrics.compute_coverage_and_missing(X)

            # Compute cardinality
            unique_values = audit_metrics.compute_cardinality(X)

            # Compute variance (numeric only)
            variance = audit_metrics.compute_variance_safe(X)

            # Build DataFrame
            stats_df = pd.DataFrame({
                'feature_name': X.columns,
                'data_type': [feature_types[col] for col in X.columns],
                'coverage': coverage.values,
                'missing_rate': missing_rate.values,
                'unique_values': unique_values.values,
                'variance': variance.values,
                'cardinality': unique_values.values,  # Alias for categorical analysis
                'categorical_flag': [col in categorical_features for col in X.columns]
            })

            return stats_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to compute basic stats",
                error_message=str(e),
                traceback=traceback.format_exc()
            )

    def compute_importance_metrics(self,
                                   results: ModelTrainingResults,
                                   X_eval: pd.DataFrame,
                                   y_eval: pd.Series) -> pd.DataFrame:
        """Compute feature importance metrics."""
        try:
            importance_df = pd.DataFrame({'feature_name': results.feature_names})

            # SHAP importance (reuse from results if available)
            if self.audit_config.compute_shap_importance and results.shap_values is not None:
                shap_df = audit_metrics.compute_shap_statistics(
                    results.shap_values,
                    results.feature_names
                )
                importance_df = importance_df.merge(shap_df, on='feature_name', how='left')
            else:
                importance_df['shap_mean_abs'] = 0.0
                importance_df['shap_std_abs'] = 0.0
                importance_df['shap_rank'] = 0

            # Permutation importance
            if self.audit_config.compute_permutation_importance:
                perm_df = audit_metrics.compute_permutation_importance_safe(
                    results.model,
                    X_eval,
                    y_eval,
                    n_repeats=self.audit_config.permutation_n_repeats
                )
                importance_df = importance_df.merge(perm_df, on='feature_name', how='left')
            else:
                importance_df['permutation_importance_mean'] = 0.0
                importance_df['permutation_importance_std'] = 0.0

            # Model-specific importance
            model_importance = audit_metrics.extract_model_importance(
                results.model,
                results.feature_names
            )
            if model_importance is not None:
                importance_df['model_gain_importance'] = model_importance
            else:
                importance_df['model_gain_importance'] = 0.0

            return importance_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to compute importance metrics",
                error_message=str(e),
                traceback=traceback.format_exc()
            )

    def compute_redundancy_metrics(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute feature redundancy metrics."""
        try:
            redundancy_df = pd.DataFrame({'feature_name': X.columns})

            # Pairwise max correlation
            pairwise_max_corr = audit_metrics.compute_pairwise_max_correlation(
                X,
                method=self.audit_config.correlation_method
            )
            redundancy_df['pairwise_max_corr'] = redundancy_df['feature_name'].map(pairwise_max_corr)

            # VIF (optional, expensive)
            if self.audit_config.compute_vif:
                vif = audit_metrics.compute_vif(X)
                redundancy_df['vif'] = redundancy_df['feature_name'].map(vif)

            return redundancy_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to compute redundancy metrics",
                error_message=str(e),
                traceback=traceback.format_exc()
            )

    def compute_stability_scores(self,
                                 results: ModelTrainingResults,
                                 fold_importances: Optional[List[Dict[str, float]]] = None) -> pd.DataFrame:
        """Compute stability scores across CV folds."""
        try:
            # Use fold_importances from results if not provided
            if fold_importances is None and hasattr(results, 'fold_importances'):
                fold_importances = results.fold_importances

            # Check if we have fold data
            if not fold_importances or len(fold_importances) == 0:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No fold importances available for stability analysis",
                    n_folds=results.n_folds,
                    has_fold_importances=hasattr(results, 'fold_importances')
                )
                # Return NaN to indicate unavailable (not 1.0 placeholder)
                return pd.DataFrame({
                    'feature_name': results.feature_names,
                    'stability_score': np.nan,
                    'stability_score_shap': np.nan
                })

            # Compute gain importance stability
            stability_scores = audit_metrics.compute_stability_from_fold_importances(
                fold_importances,
                results.feature_names,
                top_k=self.audit_config.stability_top_k
            )

            # Compute SHAP importance stability if available
            shap_stability_scores = pd.Series(np.nan, index=results.feature_names)
            if hasattr(results, 'fold_shap_importances') and len(results.fold_shap_importances) > 0:
                shap_stability_scores = audit_metrics.compute_stability_from_fold_importances(
                    results.fold_shap_importances,
                    results.feature_names,
                    top_k=self.audit_config.stability_top_k
                )

                self.app_logger.structured_log(
                    logging.INFO,
                    "Computed SHAP stability scores",
                    n_folds=len(results.fold_shap_importances),
                    mean_shap_stability=float(shap_stability_scores.mean())
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Computed stability scores",
                n_folds=len(fold_importances),
                top_k=self.audit_config.stability_top_k,
                mean_stability=float(stability_scores.mean()),
                has_shap_stability=not shap_stability_scores.isna().all()
            )

            return pd.DataFrame({
                'feature_name': results.feature_names,
                'stability_score': stability_scores.values,
                'stability_score_shap': shap_stability_scores.values
            })

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to compute stability scores",
                error_message=str(e),
                traceback=traceback.format_exc()
            )

    def compute_drop_candidate_scores(self, audit_df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite drop candidate scores and flags."""
        try:
            # Near-zero importance flag
            if 'shap_mean_abs' in audit_df.columns and 'permutation_importance_mean' in audit_df.columns:
                audit_df['near_zero_importance'] = audit_metrics.flag_near_zero_importance(
                    audit_df['shap_mean_abs'],
                    audit_df['permutation_importance_mean'],
                    percentile_threshold=self.threshold_config.near_zero_importance_percentile,
                    absolute_threshold=self.threshold_config.near_zero_importance_threshold
                )
            else:
                audit_df['near_zero_importance'] = 0

            # High missing flag
            audit_df['high_missing_flag'] = (
                audit_df['missing_rate'] > self.threshold_config.high_missing_threshold
            ).astype(int)

            # High collinearity flag
            if 'pairwise_max_corr' in audit_df.columns:
                audit_df['high_collinearity_flag'] = (
                    audit_df['pairwise_max_corr'] > self.threshold_config.high_collinearity_threshold
                ).astype(int)
            else:
                audit_df['high_collinearity_flag'] = 0

            # Unstable flag (based on gain importance stability OR SHAP stability)
            if 'stability_score' in audit_df.columns:
                unstable_gain = audit_df['stability_score'] < self.threshold_config.low_stability_threshold

                # Also consider SHAP stability if available
                if 'stability_score_shap' in audit_df.columns:
                    unstable_shap = audit_df['stability_score_shap'] < self.threshold_config.low_stability_threshold
                    # Flag if unstable in EITHER metric (OR logic)
                    audit_df['unstable_flag'] = (unstable_gain | unstable_shap).astype(int)
                else:
                    audit_df['unstable_flag'] = unstable_gain.astype(int)
            else:
                audit_df['unstable_flag'] = 0

            # Leakage flag (manual annotation)
            audit_df['leakage_flag'] = 0

            # Composite drop candidate score
            audit_df['drop_candidate_score'] = audit_metrics.compute_composite_drop_score(
                audit_df['near_zero_importance'],
                audit_df['high_missing_flag'],
                audit_df['high_collinearity_flag'],
                audit_df['unstable_flag'],
                audit_df['leakage_flag']
            )

            # Add notes column for manual annotations
            audit_df['notes'] = ''

            return audit_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to compute drop candidate scores",
                error_message=str(e),
                traceback=traceback.format_exc()
            )
