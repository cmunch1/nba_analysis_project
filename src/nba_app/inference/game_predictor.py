"""Game Predictor

Orchestrates the complete inference pipeline:
1. Feature engineering for upcoming games
2. Model loading and inference
3. Postprocessing (calibration + conformal prediction)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.inference.model_predictor import ModelPredictor
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.model_registry.base_model_registry import BaseModelRegistry
from ml_framework.postprocessing.probability_calibrator import ProbabilityCalibrator
from ml_framework.postprocessing.conformal_predictor import ConformalPredictor

from nba_app.feature_engineering.feature_engineer import FeatureEngineer


class GamePredictor:
    """
    Complete prediction pipeline for NBA games.

    Integrates feature engineering, model inference, and postprocessing
    to generate production-ready predictions with uncertainty quantification.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 data_access: BaseDataAccess,
                 model_registry: BaseModelRegistry,
                 feature_engineer: FeatureEngineer,
                 model_predictor: ModelPredictor):
        """
        Initialize GamePredictor.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            data_access: Data access layer
            model_registry: Model registry for loading production model
            feature_engineer: Feature engineering component
            model_predictor: Model inference component
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.data_access = data_access
        self.model_registry = model_registry
        self.feature_engineer = feature_engineer
        self.model_predictor = model_predictor

        # Postprocessing components (loaded with model)
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.conformal_predictor: Optional[ConformalPredictor] = None

        self.app_logger.structured_log(
            logging.INFO,
            "GamePredictor initialized"
        )

    def load_production_model(self) -> None:
        """
        Load the production model and its postprocessing artifacts.

        Raises:
            Error if model cannot be loaded
        """
        try:
            model_identifier = self.config.inference.model.identifier

            self.app_logger.structured_log(
                logging.INFO,
                "Loading production model",
                model_identifier=model_identifier
            )

            # Load model via ModelPredictor (handles preprocessing automatically)
            self.model_predictor.load_model(model_identifier)

            # Load postprocessing artifacts if enabled
            if self.config.inference.postprocessing.enable_calibration:
                self._load_calibration_artifact(model_identifier)

            if self.config.inference.postprocessing.enable_conformal:
                self._load_conformal_artifact(model_identifier)

            self.app_logger.structured_log(
                logging.INFO,
                "Production model loaded successfully",
                has_calibrator=self.calibrator is not None,
                has_conformal=self.conformal_predictor is not None
            )

        except Exception as e:
            # Try fallback model if configured
            fallback = self.config.inference.model.fallback_identifier
            if fallback:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Primary model failed, trying fallback",
                    fallback_identifier=fallback,
                    error=str(e)
                )
                self.model_predictor.load_model(fallback)
            else:
                raise self.error_handler.create_error_handler(
                    'inference',
                    "Error loading production model",
                    original_error=str(e),
                    model_identifier=model_identifier
                )

    def _load_calibration_artifact(self, model_identifier: str) -> None:
        """Load calibration artifact from model registry."""
        try:
            # Load calibration artifact from registry
            model_data = self.model_registry.load_model(model_identifier)
            calibration_artifact = model_data.get('calibration_artifact')

            if calibration_artifact:
                # Restore calibrator
                self.calibrator = ProbabilityCalibrator(
                    self.config,
                    self.app_logger,
                    self.error_handler
                )
                # Assume calibrator has a load method (similar to preprocessor)
                if hasattr(self.calibrator, 'load_artifact'):
                    self.calibrator.load_artifact(calibration_artifact)

                self.app_logger.structured_log(
                    logging.INFO,
                    "Calibration artifact loaded",
                    method=calibration_artifact.get('method')
                )
            elif not self.config.inference.postprocessing.skip_if_missing:
                raise ValueError("Calibration enabled but artifact not found in model")

        except Exception as e:
            if not self.config.inference.postprocessing.skip_if_missing:
                raise
            self.app_logger.structured_log(
                logging.WARNING,
                "Calibration artifact not found, using raw probabilities",
                error=str(e)
            )

    def _load_conformal_artifact(self, model_identifier: str) -> None:
        """Load conformal prediction artifact from model registry."""
        try:
            model_data = self.model_registry.load_model(model_identifier)
            conformal_artifact = model_data.get('conformal_artifact')

            if conformal_artifact:
                self.conformal_predictor = ConformalPredictor(
                    self.config,
                    self.app_logger,
                    self.error_handler
                )
                if hasattr(self.conformal_predictor, 'load_artifact'):
                    self.conformal_predictor.load_artifact(conformal_artifact)

                self.app_logger.structured_log(
                    logging.INFO,
                    "Conformal artifact loaded",
                    method=conformal_artifact.get('method')
                )
            elif not self.config.inference.postprocessing.skip_if_missing:
                raise ValueError("Conformal prediction enabled but artifact not found")

        except Exception as e:
            if not self.config.inference.postprocessing.skip_if_missing:
                raise
            self.app_logger.structured_log(
                logging.WARNING,
                "Conformal artifact not found, skipping uncertainty quantification",
                error=str(e)
            )

    def predict_games(self, schedule_df: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for scheduled games.

        Pipeline:
        1. Engineer features from historical data
        2. Run inference (preprocessing + prediction)
        3. Apply postprocessing (calibration + conformal)
        4. Format output

        Args:
            schedule_df: Scheduled games with columns: game_id, game_date, home_team, away_team
            historical_data: Historical box scores for feature engineering

        Returns:
            DataFrame with predictions containing:
            - game_id, game_date, home_team, away_team
            - raw_home_win_prob: Raw model output
            - calibrated_home_win_prob: Calibrated probability
            - predicted_winner: 'home' or 'away'
            - prediction_set: Conformal prediction set
            - prob_lower, prob_upper: Conformal uncertainty intervals

        Raises:
            Error if prediction fails
        """
        try:
            if schedule_df.empty:
                self.app_logger.structured_log(
                    logging.INFO,
                    "No games to predict"
                )
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Starting prediction pipeline",
                num_games=len(schedule_df)
            )

            # Step 1: Feature engineering
            features_df = self._engineer_features(schedule_df, historical_data)

            # Step 2: Inference (preprocessing happens automatically in ModelPredictor)
            raw_probabilities = self._run_inference(features_df)

            # Step 3: Postprocessing
            calibrated_probs, prediction_sets, intervals = self._apply_postprocessing(
                raw_probabilities,
                features_df
            )

            # Step 4: Format output
            predictions_df = self._format_predictions(
                schedule_df,
                raw_probabilities,
                calibrated_probs,
                prediction_sets,
                intervals
            )

            self.app_logger.structured_log(
                logging.INFO,
                "Prediction pipeline completed",
                num_predictions=len(predictions_df),
                avg_home_win_prob=float(np.mean(calibrated_probs))
            )

            return predictions_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error in prediction pipeline",
                original_error=str(e),
                num_games=len(schedule_df) if schedule_df is not None else None
            )

    def _engineer_features(self, schedule_df: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for upcoming games using historical data.

        This reuses the training feature engineering logic but operates
        on scheduled games instead of historical games.

        Args:
            schedule_df: Scheduled games
            historical_data: Historical box scores

        Returns:
            DataFrame with engineered features matching training schema
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Engineering features for upcoming games",
                num_games=len(schedule_df),
                historical_games=len(historical_data)
            )

            # Prepare data format expected by FeatureEngineer
            # Note: This assumes your FeatureEngineer can handle both batch (training)
            # and online (inference) modes. If not, we'll need to refactor it.

            # For now, create a combined dataset with historical + scheduled games
            # and engineer features only for scheduled games

            # Create placeholder rows for scheduled games
            scheduled_games_data = []
            for _, game in schedule_df.iterrows():
                # Create rows for home and away teams
                scheduled_games_data.append({
                    'game_id': game['game_id'],
                    'game_date': game['game_date'],
                    'team': game['home_team'],
                    'opponent': game['away_team'],
                    'is_home': True
                })
                scheduled_games_data.append({
                    'game_id': game['game_id'],
                    'game_date': game['game_date'],
                    'team': game['away_team'],
                    'opponent': game['home_team'],
                    'is_home': False
                })

            scheduled_df = pd.DataFrame(scheduled_games_data)

            # Combine with historical data
            combined_df = pd.concat([historical_data, scheduled_df], ignore_index=True)
            combined_df = combined_df.sort_values('game_date')

            # Engineer features (rolling stats will be calculated from historical data)
            engineered_df = self.feature_engineer.engineer_features(
                combined_df,
                export_schema=False  # Don't export schema during inference
            )

            # Filter to only scheduled games
            features_df = engineered_df[engineered_df['game_id'].isin(schedule_df['game_id'])]

            # Merge home/away into game-centric format
            features_df = self.feature_engineer.merge_team_data(features_df)

            # Encode game date
            features_df = self.feature_engineer.encode_game_date(features_df)

            self.app_logger.structured_log(
                logging.INFO,
                "Feature engineering completed",
                num_features=len(features_df.columns),
                output_shape=features_df.shape
            )

            return features_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error engineering features",
                original_error=str(e)
            )

    def _run_inference(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Run model inference with automatic preprocessing.

        Args:
            features_df: Engineered features

        Returns:
            Raw probability predictions (home team win probability)
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Running model inference",
                input_shape=features_df.shape
            )

            # ModelPredictor.predict() automatically applies preprocessing
            probabilities = self.model_predictor.predict(
                features_df,
                return_probabilities=True
            )

            self.app_logger.structured_log(
                logging.INFO,
                "Inference completed",
                num_predictions=len(probabilities),
                prob_range=(float(np.min(probabilities)), float(np.max(probabilities)))
            )

            return probabilities

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error during model inference",
                original_error=str(e)
            )

    def _apply_postprocessing(self,
                            raw_probabilities: np.ndarray,
                            features_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[list], Optional[dict]]:
        """
        Apply calibration and conformal prediction.

        Args:
            raw_probabilities: Raw model outputs
            features_df: Features (for conformal prediction)

        Returns:
            Tuple of (calibrated_probs, prediction_sets, intervals)
        """
        try:
            calibrated_probs = raw_probabilities
            prediction_sets = None
            intervals = None

            # Apply calibration
            if self.calibrator is not None:
                self.app_logger.structured_log(
                    logging.INFO,
                    "Applying probability calibration"
                )
                calibrated_probs = self.calibrator.transform(raw_probabilities)

            # Apply conformal prediction
            if self.conformal_predictor is not None:
                self.app_logger.structured_log(
                    logging.INFO,
                    "Generating conformal prediction sets"
                )
                # Conformal predictor expects calibrated probabilities
                conformal_results = self.conformal_predictor.transform(calibrated_probs)
                prediction_sets = conformal_results.get('prediction_sets')
                intervals = conformal_results.get('probability_intervals')

            return calibrated_probs, prediction_sets, intervals

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error in postprocessing",
                original_error=str(e)
            )

    def _format_predictions(self,
                          schedule_df: pd.DataFrame,
                          raw_probs: np.ndarray,
                          calibrated_probs: np.ndarray,
                          prediction_sets: Optional[list],
                          intervals: Optional[dict]) -> pd.DataFrame:
        """
        Format predictions into output DataFrame.

        Args:
            schedule_df: Original scheduled games
            raw_probs: Raw probabilities
            calibrated_probs: Calibrated probabilities
            prediction_sets: Conformal prediction sets
            intervals: Conformal probability intervals

        Returns:
            Formatted predictions DataFrame
        """
        try:
            # Start with schedule info
            output_df = schedule_df[['game_id', 'game_date', 'home_team', 'away_team']].copy()

            # Add probabilities
            output_df['raw_home_win_prob'] = raw_probs
            output_df['calibrated_home_win_prob'] = calibrated_probs

            # Predicted winner
            output_df['predicted_winner'] = output_df['calibrated_home_win_prob'].apply(
                lambda p: 'home' if p >= 0.5 else 'away'
            )

            # Predicted probability (calibrated probability of predicted winner)
            output_df['predicted_probability'] = output_df['calibrated_home_win_prob'].apply(
                lambda p: max(p, 1 - p)
            )

            # Add conformal prediction sets
            if prediction_sets is not None:
                output_df['prediction_set'] = [str(ps) for ps in prediction_sets]

            # Add uncertainty intervals
            if intervals is not None:
                # intervals is a numpy array with shape (n, 2) where [:, 0] is lower, [:, 1] is upper
                output_df['prob_lower'] = intervals[:, 0]
                output_df['prob_upper'] = intervals[:, 1]
                output_df['interval_width'] = output_df['prob_upper'] - output_df['prob_lower']

            # Add metadata
            output_df['prediction_timestamp'] = datetime.now().isoformat()
            output_df['model_identifier'] = self.model_predictor.model_identifier

            return output_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error formatting predictions",
                original_error=str(e)
            )
