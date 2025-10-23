"""Dashboard Data Generator

Main orchestrator that aggregates all dashboard data sections into a single CSV.
"""

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

from .predictions_aggregator import PredictionsAggregator
from .results_aggregator import ResultsAggregator
from .performance_calculator import PerformanceCalculator
from .team_performance_analyzer import TeamPerformanceAnalyzer


class DashboardDataGenerator:
    """
    Orchestrates the generation of dashboard-optimized data.

    Combines predictions, results, performance metrics, and team analysis
    into a single denormalized CSV file for dashboard consumption.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 data_access: BaseDataAccess,
                 error_handler: BaseErrorHandler,
                 predictions_aggregator: PredictionsAggregator,
                 results_aggregator: ResultsAggregator,
                 performance_calculator: PerformanceCalculator,
                 team_performance_analyzer: TeamPerformanceAnalyzer):
        """
        Initialize DashboardDataGenerator.

        Args:
            config: Configuration manager
            app_logger: Application logger
            data_access: Data access layer
            error_handler: Error handler
            predictions_aggregator: Predictions aggregator component
            results_aggregator: Results aggregator component
            performance_calculator: Performance metrics calculator
            team_performance_analyzer: Team performance analyzer
        """
        self.config = config
        self.app_logger = app_logger
        self.data_access = data_access
        self.error_handler = error_handler
        self.predictions_aggregator = predictions_aggregator
        self.results_aggregator = results_aggregator
        self.performance_calculator = performance_calculator
        self.team_performance_analyzer = team_performance_analyzer

        self.app_logger.structured_log(
            logging.INFO,
            "DashboardDataGenerator initialized"
        )

    def generate_dashboard_data(self) -> pd.DataFrame:
        """
        Generate complete dashboard data from all sources.

        Returns:
            Combined DataFrame with all dashboard sections

        Raises:
            Error if critical data cannot be loaded
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Starting dashboard data generation"
            )

            all_sections = []

            # Section 1: Tomorrow's Predictions
            self.app_logger.structured_log(logging.INFO, "Processing predictions section")
            predictions_section = self._generate_predictions_section()
            if not predictions_section.empty:
                all_sections.append(predictions_section)
            else:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No predictions available for dashboard"
                )

            # Section 2: Yesterday's Results (with validation)
            self.app_logger.structured_log(logging.INFO, "Processing results section")
            results_section = self._generate_results_section()
            if not results_section.empty:
                all_sections.append(results_section)

            # Section 3: Performance Metrics
            self.app_logger.structured_log(logging.INFO, "Processing performance metrics section")
            # Need validated results for metrics
            validated_results = self._get_all_validated_results()
            if not validated_results.empty:
                metrics_section = self.performance_calculator.calculate_metrics(validated_results)
                if not metrics_section.empty:
                    all_sections.append(metrics_section)

                # Section 4: Team Performance
                self.app_logger.structured_log(logging.INFO, "Processing team performance section")
                team_section = self.team_performance_analyzer.calculate_team_metrics(validated_results)
                if not team_section.empty:
                    all_sections.append(team_section)

                # Section 5: Drift Monitoring
                self.app_logger.structured_log(logging.INFO, "Processing drift monitoring section")
                drift_section = self.performance_calculator.get_drift_data(validated_results)
                if not drift_section.empty:
                    all_sections.append(drift_section)

            # Combine all sections
            if not all_sections:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No dashboard data generated - all sections empty"
                )
                return pd.DataFrame()

            dashboard_df = pd.concat(all_sections, ignore_index=True)

            self.app_logger.structured_log(
                logging.INFO,
                "Dashboard data generation completed",
                total_rows=len(dashboard_df),
                num_sections=len(all_sections)
            )

            return dashboard_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error generating dashboard data",
                original_error=str(e)
            )

    def _generate_predictions_section(self) -> pd.DataFrame:
        """Generate predictions section for dashboard."""
        try:
            # Load tomorrow's predictions
            predictions_df = self.predictions_aggregator.load_predictions()

            if predictions_df.empty:
                return pd.DataFrame()

            # Format for dashboard
            formatted_df = self.predictions_aggregator.format_for_dashboard(predictions_df)

            return formatted_df

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Error generating predictions section",
                error=str(e)
            )
            return pd.DataFrame()

    def _generate_results_section(self) -> pd.DataFrame:
        """Generate results section for dashboard."""
        try:
            # Load recent results
            results_df = self.results_aggregator.load_recent_results()

            if results_df.empty:
                return pd.DataFrame()

            # Load corresponding predictions
            # Look back over the configured lookback period
            lookback_days = self.config.dashboard_prep.results.lookback_days
            predictions_list = []

            for i in range(1, lookback_days + 1):
                date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                preds = self.predictions_aggregator.load_predictions(prediction_date=date)
                if not preds.empty:
                    predictions_list.append(preds)

            if not predictions_list:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No historical predictions found for results validation"
                )
                return pd.DataFrame()

            all_predictions = pd.concat(predictions_list, ignore_index=True)

            # Validate results against predictions
            validated_df = self.results_aggregator.validate_against_predictions(
                results_df,
                all_predictions
            )

            if validated_df.empty:
                return pd.DataFrame()

            # Format for dashboard
            formatted_df = self.results_aggregator.format_for_dashboard(validated_df)

            return formatted_df

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Error generating results section",
                error=str(e)
            )
            return pd.DataFrame()

    def _get_all_validated_results(self) -> pd.DataFrame:
        """
        Get all validated results for metrics calculation.

        Loads all historical results and matches with predictions.

        Returns:
            DataFrame with all validated results
        """
        try:
            # Load all results from season start
            season_start = self.config.dashboard_prep.performance.time_windows[0].get('start_date')

            if season_start:
                # Calculate days since season start
                season_start_date = pd.to_datetime(season_start)
                days_since_start = (datetime.now() - season_start_date).days
            else:
                # Default to 120 days
                days_since_start = 120

            self.app_logger.structured_log(
                logging.INFO,
                "Loading all results for metrics calculation",
                lookback_days=days_since_start
            )

            results_df = self.results_aggregator.load_recent_results(lookback_days=days_since_start)

            if results_df.empty:
                return pd.DataFrame()

            # Load all corresponding predictions
            predictions_list = []

            for i in range(1, days_since_start + 1):
                date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                preds = self.predictions_aggregator.load_predictions(prediction_date=date)
                if not preds.empty:
                    predictions_list.append(preds)

            if not predictions_list:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No predictions found for metrics calculation"
                )
                return pd.DataFrame()

            all_predictions = pd.concat(predictions_list, ignore_index=True)

            # Validate
            validated_df = self.results_aggregator.validate_against_predictions(
                results_df,
                all_predictions
            )

            self.app_logger.structured_log(
                logging.INFO,
                "Validated results loaded for metrics",
                num_validated=len(validated_df)
            )

            return validated_df

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Error loading validated results",
                error=str(e)
            )
            return pd.DataFrame()

    def save_dashboard_data(self, dashboard_df: pd.DataFrame) -> str:
        """
        Save dashboard data to configured output location.

        Args:
            dashboard_df: Dashboard data DataFrame

        Returns:
            Path to saved file

        Raises:
            Error if save fails
        """
        try:
            if dashboard_df.empty:
                raise ValueError("Cannot save empty dashboard data")

            # Create output directory
            output_dir = Path(self.config.dashboard_prep.output.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save main dashboard file
            output_filename = self.config.dashboard_prep.output.output_filename
            output_path = output_dir / output_filename

            self.app_logger.structured_log(
                logging.INFO,
                "Saving dashboard data",
                output_path=str(output_path),
                num_rows=len(dashboard_df)
            )

            self.data_access.save_dataframes([dashboard_df], [str(output_path)])

            # Archive snapshot if enabled
            if self.config.dashboard_prep.output.archive_snapshots:
                self._save_snapshot(dashboard_df)

            self.app_logger.structured_log(
                logging.INFO,
                "Dashboard data saved successfully",
                output_path=str(output_path)
            )

            return str(output_path)

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                "Error saving dashboard data",
                original_error=str(e)
            )

    def _save_snapshot(self, dashboard_df: pd.DataFrame) -> None:
        """
        Save an archived snapshot of dashboard data.

        Args:
            dashboard_df: Dashboard data to archive
        """
        try:
            snapshot_dir = Path(self.config.dashboard_prep.output.snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            today = datetime.now().strftime('%Y-%m-%d')
            snapshot_pattern = self.config.dashboard_prep.output.snapshot_filename_pattern
            snapshot_filename = snapshot_pattern.replace('{date}', today)
            snapshot_path = snapshot_dir / snapshot_filename

            self.data_access.save_dataframes([dashboard_df], [str(snapshot_path)])

            self.app_logger.structured_log(
                logging.INFO,
                "Dashboard snapshot saved",
                snapshot_path=str(snapshot_path)
            )

        except Exception as e:
            # Don't fail main save if snapshot fails
            self.app_logger.structured_log(
                logging.WARNING,
                "Error saving dashboard snapshot",
                error=str(e)
            )
