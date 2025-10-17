"""
Calibration visualization charts for probability calibration analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from .base_chart import BaseChart
from .chart_utils import ChartUtils
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
import logging


class CalibrationCharts(BaseChart):
    """Charts for visualizing probability calibration quality."""

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize calibration charts with dependencies.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        super().__init__(config, app_logger, error_handler)
        self.chart_utils = ChartUtils(app_logger, error_handler)

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def create_figure(self,
                     calibration_data: Dict[str, Any],
                     **kwargs) -> plt.Figure:
        """
        Create calibration curve visualization.

        Args:
            calibration_data: Dictionary containing calibration curve data
            **kwargs: Additional chart parameters

        Returns:
            Calibration curve figure
        """
        return self.create_calibration_curve(calibration_data)

    @log_performance
    def create_calibration_curve(self,
                                calibration_data: Dict[str, Any]) -> plt.Figure:
        """
        Create a calibration curve (reliability diagram) showing
        predicted probabilities vs actual outcomes.

        Args:
            calibration_data: Dictionary containing:
                - uncalibrated: Dict with 'prob_pred' and 'prob_true'
                - calibrated: Dict with 'prob_pred' and 'prob_true'
                - n_bins: Number of bins used

        Returns:
            Calibration curve figure
        """
        try:
            uncal = calibration_data['uncalibrated']
            cal = calibration_data['calibrated']
            n_bins = calibration_data.get('n_bins', 10)

            fig, ax = self.chart_utils.create_figure(figsize=(10, 10))

            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

            # Plot uncalibrated predictions
            ax.plot(
                uncal['prob_pred'],
                uncal['prob_true'],
                marker='o',
                linewidth=2,
                label='Uncalibrated',
                color='#e74c3c'
            )

            # Plot calibrated predictions
            ax.plot(
                cal['prob_pred'],
                cal['prob_true'],
                marker='s',
                linewidth=2,
                label='Calibrated',
                color='#2ecc71'
            )

            ax.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax.set_ylabel('Fraction of Positives', fontsize=12)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper left', fontsize=11)

            self.app_logger.structured_log(
                logging.INFO,
                "Calibration curve created",
                n_bins=n_bins
            )

            return self.chart_utils.finalize_plot(
                fig,
                'Calibration Curve (Reliability Diagram)'
            )

        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "calibration curve",
                n_bins=calibration_data.get('n_bins') if 'calibration_data' in locals() else None
            )

    @log_performance
    def create_calibration_comparison(self,
                                     calibration_metrics: Dict[str, float]) -> plt.Figure:
        """
        Create a bar chart comparing calibration metrics before and after calibration.

        Args:
            calibration_metrics: Dictionary containing:
                - brier_score_before: Brier score before calibration
                - brier_score_after: Brier score after calibration
                - log_loss_before: Log loss before calibration
                - log_loss_after: Log loss after calibration
                - ece_before: Expected Calibration Error before
                - ece_after: Expected Calibration Error after

        Returns:
            Calibration metrics comparison figure
        """
        try:
            metrics = ['Brier Score', 'Log Loss', 'ECE']
            before_values = [
                calibration_metrics['brier_score_before'],
                calibration_metrics['log_loss_before'],
                calibration_metrics['ece_before']
            ]
            after_values = [
                calibration_metrics['brier_score_after'],
                calibration_metrics['log_loss_after'],
                calibration_metrics['ece_after']
            ]

            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = self.chart_utils.create_figure(figsize=(10, 6))

            bars1 = ax.bar(
                x - width/2,
                before_values,
                width,
                label='Before Calibration',
                color='#e74c3c',
                alpha=0.8
            )
            bars2 = ax.bar(
                x + width/2,
                after_values,
                width,
                label='After Calibration',
                color='#2ecc71',
                alpha=0.8
            )

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

            ax.set_ylabel('Score (Lower is Better)', fontsize=12)
            ax.set_xlabel('Metric', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend(fontsize=11)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            self.app_logger.structured_log(
                logging.INFO,
                "Calibration comparison chart created",
                brier_improvement=calibration_metrics.get('brier_score_improvement'),
                ece_improvement=calibration_metrics.get('ece_improvement')
            )

            return self.chart_utils.finalize_plot(
                fig,
                'Calibration Metrics Comparison'
            )

        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "calibration metrics comparison"
            )

    @log_performance
    def create_probability_histogram(self,
                                    y_pred_uncalibrated: np.ndarray,
                                    y_pred_calibrated: np.ndarray,
                                    y_true: np.ndarray,
                                    bins: int = 20) -> plt.Figure:
        """
        Create histograms comparing predicted probability distributions
        before and after calibration, split by true class.

        Args:
            y_pred_uncalibrated: Uncalibrated predicted probabilities
            y_pred_calibrated: Calibrated predicted probabilities
            y_true: True labels
            bins: Number of histogram bins

        Returns:
            Probability distribution histogram figure
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Split by true class
            pos_mask = y_true == 1
            neg_mask = y_true == 0

            # Uncalibrated histogram
            ax1.hist(
                y_pred_uncalibrated[pos_mask],
                bins=bins,
                alpha=0.6,
                label='True Positive',
                color='#2ecc71',
                edgecolor='black'
            )
            ax1.hist(
                y_pred_uncalibrated[neg_mask],
                bins=bins,
                alpha=0.6,
                label='True Negative',
                color='#e74c3c',
                edgecolor='black'
            )
            ax1.set_xlabel('Predicted Probability', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Uncalibrated Probabilities', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

            # Calibrated histogram
            ax2.hist(
                y_pred_calibrated[pos_mask],
                bins=bins,
                alpha=0.6,
                label='True Positive',
                color='#2ecc71',
                edgecolor='black'
            )
            ax2.hist(
                y_pred_calibrated[neg_mask],
                bins=bins,
                alpha=0.6,
                label='True Negative',
                color='#e74c3c',
                edgecolor='black'
            )
            ax2.set_xlabel('Predicted Probability', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Calibrated Probabilities', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

            plt.tight_layout()

            self.app_logger.structured_log(
                logging.INFO,
                "Probability histogram created",
                n_samples=len(y_true),
                n_bins=bins
            )

            return self.chart_utils.finalize_plot(
                fig,
                'Probability Distribution Comparison'
            )

        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "probability histogram",
                n_samples=len(y_true) if 'y_true' in locals() else None
            )
