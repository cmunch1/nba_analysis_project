"""
Feature selection module for selecting the most relevant features for the model.

This module includes methods for:
    - Correlation analysis
    - Feature importance calculation
    - Variance threshold
    - Recursive feature elimination
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging import log_performance, structured_log
from ml_framework.core.error_handling.error_handler import FeatureSelectionError
from .base_feature_engineering import BaseFeatureSelector

logger = logging.getLogger(__name__)

class FeatureSelector(BaseFeatureSelector):
    @log_performance
    def __init__(self, config: BaseConfigManager):
        """
        Initialize the FeatureSelector class.

        Args:
            config (BaseConfigManager): Configuration object containing feature selection parameters.
        """
        self.config = config
        
        structured_log(logger, logging.INFO, "FeatureSelector initialized",
                       config_type=type(config).__name__)

    @log_performance
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature selection on the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with selected features.

        Raises:
            FeatureSelectionError: If there's an error during feature selection.
        """
        structured_log(logger, logging.INFO, "Starting feature selection",
                       input_shape=df.shape)
        try:
            # Implement feature selection steps here
            df = self._remove_unnecessary_features(df)
            df = self._remove_elo_data_leakage_features(df)
            #df = self._remove_low_variance_features(df)
            #df = self._remove_highly_correlated_features(df)
            #df = self._select_features_by_importance(df)

            structured_log(logger, logging.INFO, "Feature selection completed",
                           output_shape=df.shape)
            
            return df
        
        except Exception as e:
            raise FeatureSelectionError("Error in feature selection",
                                        error_message=str(e),
                                        dataframe_shape=df.shape)
        
    def _remove_unnecessary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnecessary features.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with unnecessary features removed.
        """
        structured_log(logger, logging.INFO, "Removing unnecessary features")

        try:
            df = df.drop(columns=[f"{self.config.home_team_prefix}{col}" for col in self.config.non_useful_columns])
            df = df.drop(columns=[f"{self.config.visitor_team_prefix}{col}" for col in self.config.non_useful_columns])
            df = df.drop(columns=[self.config.new_date_column])

            structured_log(logger, logging.INFO, "Unnecessary features removed",
                           output_shape=df.shape)
            
            return df
        except Exception as e:
            raise FeatureSelectionError("Error in removing unnecessary features",
                                        error_message=str(e),
                                        dataframe_shape=df.shape)
        
    def _remove_elo_data_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ELO features that are data leakage.
        elo_team_after, elo_change are post-game columns and should not be used to predict the outcome of a game.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with ELO features removed.
        """
        structured_log(logger, logging.INFO, "Removing ELO features")

        elo_data_leakage_columns = [
            self.config.team_elo_after_column,
            self.config.elo_change_column
        ]

        try:
            df = df.drop(columns=[f"{self.config.home_team_prefix}{col}" for col in elo_data_leakage_columns])
            df = df.drop(columns=[f"{self.config.visitor_team_prefix}{col}" for col in elo_data_leakage_columns])

            structured_log(logger, logging.INFO, "ELO features removed",
                           output_shape=df.shape)
            
            return df
        except Exception as e:
            raise FeatureSelectionError("Error in removing ELO features",
                                        error_message=str(e),
                                        dataframe_shape=df.shape)
    
    def _remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with low variance.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with low variance features removed.
        """
        structured_log(logger, logging.INFO, "Removing low variance features")
        
        # Implement low variance feature removal logic here
        
        return df

    def _remove_highly_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with highly correlated features removed.
        """
        structured_log(logger, logging.INFO, "Removing highly correlated features")
        
        # Implement correlation analysis and feature removal logic here
        
        return df

    def _select_features_by_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on importance scores.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with selected important features.
        """
        structured_log(logger, logging.INFO, "Selecting features by importance")
        
        # Implement feature importance calculation and selection logic here
        
        return df

    @log_performance
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance scores.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target series.

        Returns:
            Dict[str, float]: Dictionary of feature importance scores.

        Raises:
            FeatureSelectionError: If there's an error during feature importance calculation.
        """
        structured_log(logger, logging.INFO, "Calculating feature importance")
        try:
            # Implement feature importance calculation logic here
            
            return {}  # Replace with actual feature importance dictionary
        except Exception as e:
            raise FeatureSelectionError("Error in feature importance calculation",
                                        error_message=str(e),
                                        dataframe_shape=X.shape)

    @log_performance
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int) -> List[str]:
        """
        Perform recursive feature elimination.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target series.
            n_features_to_select (int): Number of features to select.

        Returns:
            List[str]: List of selected feature names.

        Raises:
            FeatureSelectionError: If there's an error during recursive feature elimination.
        """
        structured_log(logger, logging.INFO, "Performing recursive feature elimination",
                       n_features_to_select=n_features_to_select)
        try:
            # Implement recursive feature elimination logic here
            
            return []  # Replace with actual list of selected features
        except Exception as e:
            raise FeatureSelectionError("Error in recursive feature elimination",
                                        error_message=str(e),
                                        dataframe_shape=X.shape)
        
    @log_performance
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataframe into training and validation sets.
        The validation set will be selected from the most recent n completed seasons.
        (e.g. we will pull off the last n seasons, split off a portion for validation, then add the remainder back to the original dataframe)

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
        """ 
        structured_log(logger, logging.INFO, "Splitting data into training and validation sets")
        try:

            # determine the last season in the dataframe    
            last_season = df[self.config.season_column].max()

            # determine the start season for the validation set
            validation_start_season = last_season - self.config.validation_last_n_seasons

            # limit the working dataframe to the last n seasons
            working_df = df[df[self.config.season_column] >= validation_start_season]
            df = df.drop(working_df.index)

            # use a stratified split to ensure that seasonality is maintained in the validation set (e.g. same number of games from each month)
            training_df, validation_df = train_test_split(
                working_df, 
                test_size=self.config.validation_split, 
                stratify=working_df[self.config.stratify_column],
                random_state=self.config.random_state
            )

            training_df = pd.concat([df, training_df]) 

            structured_log(logger, logging.INFO, "Data split completed",
                           training_shape=training_df.shape,
                           validation_shape=validation_df.shape)
            
            return training_df, validation_df  
        except Exception as e:
            raise FeatureSelectionError("Error in splitting data",
                                        error_message=str(e),
                                        dataframe_shape=df.shape)
    

