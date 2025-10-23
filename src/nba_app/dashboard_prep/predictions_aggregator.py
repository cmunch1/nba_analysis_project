"""Predictions Aggregator

Loads and formats predictions for tomorrow's games for dashboard display.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class PredictionsAggregator:
    """
    Aggregates predictions for upcoming games.

    Loads predictions from inference output and formats them for dashboard display.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 data_access: BaseDataAccess,
                 error_handler: BaseErrorHandler):
        """
        Initialize PredictionsAggregator.

        Args:
            config: Configuration manager
            app_logger: Application logger
            data_access: Data access layer
            error_handler: Error handler
        """
        self.config = config
        self.app_logger = app_logger
        self.data_access = data_access
        self.error_handler = error_handler

        self.app_logger.structured_log(
            logging.INFO,
            "PredictionsAggregator initialized"
        )

    def load_predictions(self, prediction_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load predictions for a specific date.

        Args:
            prediction_date: Date string (YYYY-MM-DD). If None, uses tomorrow.

        Returns:
            DataFrame with predictions

        Raises:
            Error if predictions file not found
        """
        try:
            if prediction_date is None:
                # Default to tomorrow's predictions
                tomorrow = datetime.now() + timedelta(days=1)
                prediction_date = tomorrow.strftime('%Y-%m-%d')

            self.app_logger.structured_log(
                logging.INFO,
                "Loading predictions",
                prediction_date=prediction_date
            )

            # Build predictions file path
            predictions_dir = Path(self.config.dashboard_prep.input_paths.predictions_dir)
            filename_pattern = self.config.dashboard_prep.input_paths.predictions_pattern
            filename = filename_pattern.replace('{date}', prediction_date)
            predictions_path = predictions_dir / filename

            # Load predictions
            predictions_df = self.data_access.load_dataframe(str(predictions_path))

            if predictions_df.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No predictions found for date",
                    prediction_date=prediction_date
                )
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Predictions loaded successfully",
                num_predictions=len(predictions_df),
                prediction_date=prediction_date
            )

            return predictions_df

        except Exception as e:
            # Don't raise error - dashboard can work without predictions
            self.app_logger.structured_log(
                logging.WARNING,
                "Could not load predictions",
                prediction_date=prediction_date,
                error=str(e)
            )
            return pd.DataFrame()

    def format_for_dashboard(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format predictions for dashboard display.

        Args:
            predictions_df: Raw predictions DataFrame

        Returns:
            Formatted DataFrame with section tag and selected columns
        """
        try:
            if predictions_df.empty:
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Formatting predictions for dashboard",
                num_predictions=len(predictions_df)
            )

            # Select columns for dashboard
            include_columns = self.config.dashboard_prep.predictions.include_columns

            # Keep only columns that exist in the predictions
            available_columns = [col for col in include_columns if col in predictions_df.columns]

            if not available_columns:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No configured columns found in predictions",
                    expected_columns=include_columns,
                    available_columns=predictions_df.columns.tolist()
                )
                # Use all columns if none of the configured ones exist
                dashboard_df = predictions_df.copy()
            else:
                dashboard_df = predictions_df[available_columns].copy()

            # Add section identifier for dashboard
            dashboard_df['section'] = 'predictions'

            # Add high-confidence flag
            confidence_threshold = self.config.dashboard_prep.predictions.high_confidence_threshold
            if 'confidence' in dashboard_df.columns:
                dashboard_df['high_confidence'] = dashboard_df['confidence'] >= confidence_threshold
            elif 'calibrated_home_win_prob' in dashboard_df.columns:
                # Calculate confidence from probability
                dashboard_df['confidence'] = dashboard_df['calibrated_home_win_prob'].apply(
                    lambda p: max(p, 1 - p)
                )
                dashboard_df['high_confidence'] = dashboard_df['confidence'] >= confidence_threshold

            # Sort by game date, then by confidence (high to low)
            if 'game_date' in dashboard_df.columns and 'confidence' in dashboard_df.columns:
                dashboard_df = dashboard_df.sort_values(
                    ['game_date', 'confidence'],
                    ascending=[True, False]
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Predictions formatted for dashboard",
                num_rows=len(dashboard_df),
                num_columns=len(dashboard_df.columns)
            )

            return dashboard_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error formatting predictions for dashboard",
                original_error=str(e)
            )

    def get_predictions_summary(self, predictions_df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for predictions.

        Args:
            predictions_df: Predictions DataFrame

        Returns:
            Dictionary with summary stats
        """
        if predictions_df.empty:
            return {
                'num_games': 0,
                'avg_confidence': None,
                'home_win_percentage': None,
                'high_confidence_count': 0
            }

        summary = {
            'num_games': len(predictions_df)
        }

        if 'confidence' in predictions_df.columns:
            summary['avg_confidence'] = float(predictions_df['confidence'].mean())

        if 'predicted_winner' in predictions_df.columns:
            summary['home_win_percentage'] = float(
                (predictions_df['predicted_winner'] == 'home').mean() * 100
            )

        if 'high_confidence' in predictions_df.columns:
            summary['high_confidence_count'] = int(predictions_df['high_confidence'].sum())

        return summary
