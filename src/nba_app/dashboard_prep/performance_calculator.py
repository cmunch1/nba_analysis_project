"""Performance Calculator

Calculates model performance metrics across different time windows.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class PerformanceCalculator:
    """
    Calculates model performance metrics for different time windows.

    Computes accuracy, Brier score, log loss, calibration error, etc.
    across season, 7-day, and 30-day windows.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize PerformanceCalculator.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

        self.app_logger.structured_log(
            logging.INFO,
            "PerformanceCalculator initialized"
        )

    def calculate_metrics(self, validated_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics across all configured time windows.

        Args:
            validated_results: DataFrame with actual results and predictions

        Returns:
            DataFrame with metrics for each time window
        """
        try:
            if validated_results.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No validated results - cannot calculate metrics"
                )
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Calculating performance metrics",
                num_results=len(validated_results)
            )

            metrics_list = []

            # Get time windows from config
            time_windows = self.config.dashboard_prep.performance.time_windows

            for window in time_windows:
                window_name = window['name']
                window_data = self._filter_to_window(validated_results, window)

                if window_data.empty:
                    self.app_logger.structured_log(
                        logging.WARNING,
                        "No data for time window",
                        window_name=window_name
                    )
                    continue

                window_metrics = self._calculate_window_metrics(window_data, window_name)
                metrics_list.append(window_metrics)

            # Combine all metrics
            metrics_df = pd.DataFrame(metrics_list)

            # Add section identifier
            metrics_df['section'] = 'metrics'

            self.app_logger.structured_log(
                logging.INFO,
                "Performance metrics calculated",
                num_windows=len(metrics_df)
            )

            return metrics_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error calculating performance metrics",
                original_error=str(e)
            )

    def _filter_to_window(self, df: pd.DataFrame, window_config: dict) -> pd.DataFrame:
        """
        Filter DataFrame to a specific time window.

        Args:
            df: Full results DataFrame
            window_config: Window configuration dict

        Returns:
            Filtered DataFrame
        """
        try:
            window_name = window_config['name']

            if 'start_date' in window_config:
                # Absolute date window (e.g., season)
                start_date = window_config['start_date']
                if 'game_date' in df.columns:
                    filtered_df = df[df['game_date'] >= start_date]
                else:
                    filtered_df = df
            elif 'days_back' in window_config:
                # Relative date window (e.g., 7 days)
                days_back = window_config['days_back']
                cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                if 'game_date' in df.columns:
                    filtered_df = df[df['game_date'] >= cutoff_date]
                else:
                    filtered_df = df
            else:
                # No filter - use all data
                filtered_df = df

            return filtered_df

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Error filtering to time window",
                window_name=window_name,
                error=str(e)
            )
            return pd.DataFrame()

    def _calculate_window_metrics(self, window_data: pd.DataFrame, window_name: str) -> dict:
        """
        Calculate all metrics for a specific time window.

        Args:
            window_data: Data for this window
            window_name: Name of the window

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'window': window_name
        }

        # Get configured metrics
        metric_configs = self.config.dashboard_prep.performance.metrics

        for metric_config in metric_configs:
            metric_name = metric_config['name']
            metric_value = self._calculate_single_metric(window_data, metric_name)

            # Format based on config
            if metric_value is not None:
                format_type = metric_config.get('format', 'decimal')
                if format_type == 'percentage':
                    metrics[metric_name] = f"{metric_value * 100:.1f}%"
                elif format_type == 'integer':
                    metrics[metric_name] = int(metric_value)
                elif format_type == 'decimal':
                    decimals = metric_config.get('decimals', 3)
                    metrics[metric_name] = round(metric_value, decimals)
                else:
                    metrics[metric_name] = metric_value
            else:
                metrics[metric_name] = None

        return metrics

    def _calculate_single_metric(self, data: pd.DataFrame, metric_name: str) -> float:
        """
        Calculate a single metric.

        Args:
            data: Data to calculate metric on
            metric_name: Name of metric to calculate

        Returns:
            Metric value or None if cannot be calculated
        """
        try:
            if metric_name == 'accuracy':
                if 'prediction_correct' in data.columns:
                    return float(data['prediction_correct'].mean())

            elif metric_name == 'brier_score':
                if 'calibrated_home_win_prob' in data.columns and 'actual_home_win' in data.columns:
                    return float(np.mean((data['calibrated_home_win_prob'] - data['actual_home_win']) ** 2))

            elif metric_name == 'log_loss':
                if 'calibrated_home_win_prob' in data.columns and 'actual_home_win' in data.columns:
                    # Clip probabilities to avoid log(0)
                    probs = np.clip(data['calibrated_home_win_prob'], 1e-15, 1 - 1e-15)
                    actual = data['actual_home_win']
                    return float(-np.mean(actual * np.log(probs) + (1 - actual) * np.log(1 - probs)))

            elif metric_name == 'calibration_error':
                if 'calibrated_home_win_prob' in data.columns and 'actual_home_win' in data.columns:
                    return self._calculate_expected_calibration_error(
                        data['calibrated_home_win_prob'],
                        data['actual_home_win']
                    )

            elif metric_name == 'games_predicted':
                return float(len(data))

            elif metric_name == 'avg_predicted_probability':
                return float(data['predicted_probability'].mean())

            self.app_logger.structured_log(
                logging.WARNING,
                "Metric calculation not implemented or data missing",
                metric_name=metric_name
            )
            return None

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Error calculating metric",
                metric_name=metric_name,
                error=str(e)
            )
            return None

    def _calculate_expected_calibration_error(self,
                                             predicted_probs: pd.Series,
                                             actual_outcomes: pd.Series,
                                             n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Args:
            predicted_probs: Predicted probabilities
            actual_outcomes: Actual binary outcomes
            n_bins: Number of bins for calibration

        Returns:
            ECE value
        """
        try:
            # Create bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(predicted_probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            ece = 0.0
            total_samples = len(predicted_probs)

            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                if not bin_mask.any():
                    continue

                bin_probs = predicted_probs[bin_mask]
                bin_actuals = actual_outcomes[bin_mask]

                bin_size = len(bin_probs)
                avg_predicted_prob = bin_probs.mean()
                avg_actual_freq = bin_actuals.mean()

                ece += (bin_size / total_samples) * abs(avg_predicted_prob - avg_actual_freq)

            return float(ece)

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Error calculating ECE",
                error=str(e)
            )
            return None

    def get_drift_data(self, validated_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily metrics for drift monitoring.

        Args:
            validated_results: DataFrame with validated results

        Returns:
            DataFrame with daily metrics
        """
        try:
            if validated_results.empty or 'game_date' not in validated_results.columns:
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Calculating drift monitoring data"
            )

            # Group by date and calculate metrics
            daily_metrics = []

            for date, date_data in validated_results.groupby('game_date'):
                if len(date_data) == 0:
                    continue

                metrics = {
                    'date': date,
                    'accuracy': float(date_data['prediction_correct'].mean()) if 'prediction_correct' in date_data.columns else None,
                    'avg_predicted_probability': float(date_data['predicted_probability'].mean()) if 'predicted_probability' in date_data.columns else None
                }

                # Calculate Brier score if available
                if 'calibrated_home_win_prob' in date_data.columns and 'actual_home_win' in date_data.columns:
                    metrics['brier_score'] = float(
                        np.mean((date_data['calibrated_home_win_prob'] - date_data['actual_home_win']) ** 2)
                    )

                    # Calculate calibration error
                    metrics['calibration_error'] = self._calculate_expected_calibration_error(
                        date_data['calibrated_home_win_prob'],
                        date_data['actual_home_win']
                    )

                daily_metrics.append(metrics)

            drift_df = pd.DataFrame(daily_metrics)

            # Add section identifier
            drift_df['section'] = 'drift'

            # Sort by date
            drift_df = drift_df.sort_values('date')

            self.app_logger.structured_log(
                logging.INFO,
                "Drift monitoring data calculated",
                num_days=len(drift_df)
            )

            return drift_df

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Error calculating drift data",
                error=str(e)
            )
            return pd.DataFrame()
