"""Team Performance Analyzer

Calculates model performance metrics for each team.
"""

import logging
import pandas as pd
from typing import Optional

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class TeamPerformanceAnalyzer:
    """
    Analyzes model performance on a per-team basis.

    Calculates accuracy, confidence, and prediction counts for each team,
    separated by home/away games.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize TeamPerformanceAnalyzer.

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
            "TeamPerformanceAnalyzer initialized"
        )

    def calculate_team_metrics(self, validated_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for each team.

        Args:
            validated_results: DataFrame with validated results

        Returns:
            DataFrame with metrics per team
        """
        try:
            if validated_results.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No validated results - cannot calculate team metrics"
                )
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Calculating team-level performance metrics",
                num_results=len(validated_results)
            )

            # Get minimum games threshold
            min_games = self.config.dashboard_prep.team_performance.min_games

            team_metrics_list = []

            # Get all unique teams
            teams = set()
            if 'home_team' in validated_results.columns:
                teams.update(validated_results['home_team'].unique())
            if 'away_team' in validated_results.columns:
                teams.update(validated_results['away_team'].unique())

            for team in teams:
                # Get games for this team
                team_games = self._get_team_games(validated_results, team)

                if len(team_games) < min_games:
                    self.app_logger.structured_log(
                        logging.DEBUG,
                        "Skipping team with insufficient games",
                        team=team,
                        num_games=len(team_games),
                        min_games=min_games
                    )
                    continue

                # Calculate metrics for this team
                team_metrics = self._calculate_single_team_metrics(team, team_games, validated_results)
                team_metrics_list.append(team_metrics)

            # Create DataFrame
            team_metrics_df = pd.DataFrame(team_metrics_list)

            if not team_metrics_df.empty:
                # Sort by configured metric
                sort_by = self.config.dashboard_prep.team_performance.sort_by
                sort_order = self.config.dashboard_prep.team_performance.sort_order
                ascending = (sort_order == 'ascending')

                if sort_by in team_metrics_df.columns:
                    team_metrics_df = team_metrics_df.sort_values(sort_by, ascending=ascending)

                # Add section identifier
                team_metrics_df['section'] = 'team_performance'

            self.app_logger.structured_log(
                logging.INFO,
                "Team performance metrics calculated",
                num_teams=len(team_metrics_df)
            )

            return team_metrics_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error calculating team performance metrics",
                original_error=str(e)
            )

    def _get_team_games(self, results_df: pd.DataFrame, team: str) -> pd.DataFrame:
        """
        Get all games for a specific team.

        Args:
            results_df: Full results DataFrame
            team: Team identifier

        Returns:
            DataFrame with games involving this team
        """
        if 'home_team' in results_df.columns and 'away_team' in results_df.columns:
            team_games = results_df[
                (results_df['home_team'] == team) | (results_df['away_team'] == team)
            ]
        else:
            team_games = pd.DataFrame()

        return team_games

    def _calculate_single_team_metrics(self,
                                       team: str,
                                       team_games: pd.DataFrame,
                                       all_results: pd.DataFrame) -> dict:
        """
        Calculate metrics for a single team.

        Args:
            team: Team identifier
            team_games: Games for this team
            all_results: Full results DataFrame (for context)

        Returns:
            Dictionary with team metrics
        """
        metrics = {
            'team_name': team,
            'total_predictions': len(team_games)
        }

        # Overall accuracy
        if 'prediction_correct' in team_games.columns:
            metrics['accuracy'] = float(team_games['prediction_correct'].mean())

        # Average confidence
        if 'confidence' in team_games.columns:
            metrics['avg_confidence'] = float(team_games['confidence'].mean())
        elif 'calibrated_home_win_prob' in team_games.columns:
            confidence = team_games['calibrated_home_win_prob'].apply(lambda p: max(p, 1 - p))
            metrics['avg_confidence'] = float(confidence.mean())

        # Home/Away split
        if 'home_team' in team_games.columns:
            home_games = team_games[team_games['home_team'] == team]
            away_games = team_games[team_games['away_team'] == team]

            # Home accuracy
            if not home_games.empty and 'prediction_correct' in home_games.columns:
                metrics['home_accuracy'] = float(home_games['prediction_correct'].mean())
                metrics['home_games'] = len(home_games)
            else:
                metrics['home_accuracy'] = None
                metrics['home_games'] = 0

            # Away accuracy
            if not away_games.empty and 'prediction_correct' in away_games.columns:
                metrics['away_accuracy'] = float(away_games['prediction_correct'].mean())
                metrics['away_games'] = len(away_games)
            else:
                metrics['away_accuracy'] = None
                metrics['away_games'] = 0

        return metrics
