"""Comparison report generation for baseline vs pruned models."""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.framework.data_classes.training import ModelTrainingResults


class PruningComparison:
    """
    Generates comparison reports between baseline and pruned models.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize pruning comparison generator.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler factory
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

        self.app_logger.structured_log(
            logging.INFO,
            "PruningComparison initialized",
            config_type=type(config).__name__
        )

    def compare_runs(self,
                     baseline_results: ModelTrainingResults,
                     pruned_results: ModelTrainingResults,
                     features_dropped: List[str]) -> pd.DataFrame:
        """
        Generate comparison report between baseline and pruned runs.

        Args:
            baseline_results: Results from baseline (all features) run
            pruned_results: Results from pruned run
            features_dropped: List of features that were dropped

        Returns:
            Comparison DataFrame with metrics, deltas, and recommendation
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Generating pruning comparison report",
                n_features_dropped=len(features_dropped)
            )

            # Extract metrics
            baseline_metrics = baseline_results.metrics
            pruned_metrics = pruned_results.metrics

            # Build comparison data
            comparison_data = []

            # Metrics comparison
            metrics_to_compare = [
                ('auc', 'AUC'),
                ('accuracy', 'Accuracy'),
                ('precision', 'Precision'),
                ('recall', 'Recall'),
                ('f1_score', 'F1 Score'),
            ]

            for metric_attr, metric_name in metrics_to_compare:
                baseline_val = getattr(baseline_metrics, metric_attr, None)
                pruned_val = getattr(pruned_metrics, metric_attr, None)

                if baseline_val is not None and pruned_val is not None:
                    delta = pruned_val - baseline_val
                    pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0

                    comparison_data.append({
                        'metric': metric_name,
                        'baseline': float(baseline_val),
                        'pruned': float(pruned_val),
                        'delta': float(delta),
                        'pct_change': f"{pct_change:.2f}%"
                    })

            # Feature count comparison
            n_baseline = len(baseline_results.feature_names)
            n_pruned = len(pruned_results.feature_names)
            n_dropped = len(features_dropped)

            comparison_data.append({
                'metric': 'n_features',
                'baseline': n_baseline,
                'pruned': n_pruned,
                'delta': -n_dropped,
                'pct_change': f"{-n_dropped/n_baseline*100:.1f}%"
            })

            # Performance per feature (efficiency metric)
            auc_baseline = getattr(baseline_metrics, 'auc', 0)
            auc_pruned = getattr(pruned_metrics, 'auc', 0)

            perf_per_feature_baseline = auc_baseline / n_baseline if n_baseline > 0 else 0
            perf_per_feature_pruned = auc_pruned / n_pruned if n_pruned > 0 else 0
            perf_delta = perf_per_feature_pruned - perf_per_feature_baseline
            perf_pct_change = (perf_delta / perf_per_feature_baseline * 100) if perf_per_feature_baseline != 0 else 0

            comparison_data.append({
                'metric': 'performance_per_feature',
                'baseline': float(perf_per_feature_baseline),
                'pruned': float(perf_per_feature_pruned),
                'delta': float(perf_delta),
                'pct_change': f"{perf_pct_change:+.1f}%"
            })

            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)

            # Add recommendation
            recommendation = self._generate_recommendation(
                comparison_df,
                baseline_metrics,
                pruned_metrics
            )

            comparison_df.loc[len(comparison_df)] = {
                'metric': 'recommendation',
                'baseline': recommendation,
                'pruned': '',
                'delta': '',
                'pct_change': ''
            }

            self.app_logger.structured_log(
                logging.INFO,
                "Pruning comparison report generated",
                recommendation=recommendation,
                auc_delta=float(auc_pruned - auc_baseline) if auc_baseline and auc_pruned else None
            )

            return comparison_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Failed to generate pruning comparison",
                error_message=str(e),
                traceback=traceback.format_exc()
            )

    def _generate_recommendation(self,
                                comparison_df: pd.DataFrame,
                                baseline_metrics,
                                pruned_metrics) -> str:
        """
        Generate recommendation based on comparison results.

        Args:
            comparison_df: Comparison DataFrame
            baseline_metrics: Baseline model metrics
            pruned_metrics: Pruned model metrics

        Returns:
            Recommendation string
        """
        try:
            # Get configuration thresholds
            pruning_config = self.config.core.model_testing_config.feature_pruning
            min_auc_delta = pruning_config.min_acceptable_auc_delta
            min_acc_delta = pruning_config.min_acceptable_accuracy_delta

            # Extract deltas
            auc_row = comparison_df[comparison_df['metric'] == 'AUC']
            acc_row = comparison_df[comparison_df['metric'] == 'Accuracy']

            auc_delta = float(auc_row['delta'].values[0]) if len(auc_row) > 0 else 0
            acc_delta = float(acc_row['delta'].values[0]) if len(acc_row) > 0 else 0

            # Decision logic
            if auc_delta >= min_auc_delta and acc_delta >= min_acc_delta:
                return "ACCEPT_PRUNED"
            elif auc_delta >= min_auc_delta * 0.5:  # Within 50% of threshold
                return "CONSIDER_PRUNED"
            else:
                return "REJECT_PRUNED"

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Failed to generate recommendation, defaulting to MANUAL_REVIEW",
                error=str(e)
            )
            return "MANUAL_REVIEW"

    def save_comparison_report(self,
                               comparison_df: pd.DataFrame,
                               model_name: str,
                               eval_type: str,
                               features_dropped: List[str],
                               pruning_summary: pd.DataFrame = None) -> str:
        """
        Save comparison report to file.

        Args:
            comparison_df: Comparison DataFrame
            model_name: Model name
            eval_type: Evaluation type ('oof' or 'validation')
            features_dropped: List of features dropped
            pruning_summary: Optional pruning summary DataFrame

        Returns:
            Path to saved comparison report
        """
        try:
            # Create output directory
            output_dir = Path(self.config.core.model_testing_config.feature_pruning.comparison_report_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{model_name}_{eval_type}_pruning_comparison_{timestamp}.csv"
            file_path = output_dir / filename

            # Save main comparison
            comparison_df.to_csv(file_path, index=False)

            self.app_logger.structured_log(
                logging.INFO,
                "Pruning comparison report saved",
                file_path=str(file_path),
                n_features_dropped=len(features_dropped)
            )

            # Save pruning summary if provided
            if pruning_summary is not None:
                summary_filename = f"{model_name}_{eval_type}_pruning_summary_{timestamp}.csv"
                summary_path = output_dir / summary_filename
                pruning_summary.to_csv(summary_path, index=False)

                self.app_logger.structured_log(
                    logging.INFO,
                    "Pruning summary saved",
                    file_path=str(summary_path)
                )

            return str(file_path)

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                "Failed to save pruning comparison report",
                error_message=str(e),
                traceback=traceback.format_exc(),
                model_name=model_name
            )
