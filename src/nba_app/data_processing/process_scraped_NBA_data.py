import pandas as pd
import logging
from typing import List, Tuple, Dict
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.error_handler import DataProcessingError
from .base_data_processing_classes import BaseNBADataProcessor


class ProcessScrapedNBAData(BaseNBADataProcessor):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, app_file_handler=None):
        """
        Initialize the ProcessScrapedNBAData class.

        Args:
            config (BaseConfigManager): Configuration object containing processing parameters.
            app_logger (BaseAppLogger): Logger instance for structured logging.
            app_file_handler: File handler for saving invalid records (optional).
        """
        self.config = config
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler

        self.app_logger.structured_log(
            logging.INFO,
            "ProcessScrapedNBAData initialized",
            config_type=type(config).__name__
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def process_data(self, scraped_dataframes: List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Process the scraped NBA data through all steps.

        Args:
            scraped_dataframes (List[pd.DataFrame]): List of scraped dataframes.

        Returns:
            pd.DataFrame: Fully processed dataframe.

        Raises:
            DataProcessingError: If there's an error during any step of the data processing.
        """
        
        self.app_logger.structured_log(logging.INFO, "Starting process_data",
                       dataframe_count=len(scraped_dataframes))
        try:
                        
            df = self._merge_dataframes(scraped_dataframes)
            df = self._handle_anomalous_data(df)
            df, column_mapping = self._transform_data(df)

            df[self.config.new_date_column] = pd.to_datetime(df[self.config.new_date_column])
            df = df.sort_values(by=[self.config.new_date_column, self.config.new_game_id_column, self.config.home_team_column], ascending=[True, True, True])
                      
            self.app_logger.structured_log(logging.INFO, "Data processing completed successfully",
                           final_dataframe_shape=df.shape)
            return df, column_mapping
        
        except (DataProcessingError) as e:
            self.app_logger.structured_log(logging.ERROR, "Error in process_data",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise
        except Exception as e:
            self.app_logger.structured_log(logging.ERROR, "Unexpected error in process_data",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise DataProcessingError("Unexpected error in process_data",
                                      error_message=str(e))
        
    @log_performance
    def merge_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.app_logger.structured_log(logging.INFO, "Starting merge_team_data",
                       dataframe_shape=df.shape)
        try:
            # separate out the game info columns but keep the new_game_id_column so we can merge on it
            game_info_columns = [col for col in self.config.game_info_columns if col != self.config.new_game_id_column]
            game_info_df = df[df['is_home_team'] == 1][game_info_columns + [self.config.new_game_id_column]].copy()

            # Remove game info columns from the main dataframe, but keep new_game_id_column
            df_without_game_info = df.drop(columns=game_info_columns)

            # Split into home and away dataframes
            df_home_team = df_without_game_info[df_without_game_info[self.config.home_team_column] == 1]
            df_away_team = df_without_game_info[df_without_game_info[self.config.home_team_column] == 0]

            # Merge home and away data
            merged_df = pd.merge(df_home_team, df_away_team, on=self.config.new_game_id_column, suffixes=("_" + self.config.home_game_suffix, "_" + self.config.visitor_game_suffix))

            # Merge back the game info columns
            final_df = pd.merge(game_info_df, merged_df, on=self.config.new_game_id_column)

            # move the new_game_id_column to the front
            final_df = final_df[[self.config.new_game_id_column] + [col for col in final_df.columns if col != self.config.new_game_id_column]]

            self.app_logger.structured_log(logging.INFO, "Team data combined successfully",
                           dataframe_shape=final_df.shape)
            return final_df
        except Exception as e:
            raise DataProcessingError("Error in combine_team_data",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def validate_home_visitor_teams(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate that each game has exactly one home team and one visitor team.
        The NBA website has occasionally made errors where both teams were listed as home or visitor.

        Args:
            df (pd.DataFrame): Input dataframe with team-centric data (one row per team per game).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Valid dataframe and invalid dataframe.

        Raises:
            DataProcessingError: If there's an error during validation.
        """
        self.app_logger.structured_log(logging.INFO, "Starting validate_home_visitor_teams",
                       dataframe_shape=df.shape)
        try:
            # Use original_game_id for validation if available (before sub_season_id was stripped)
            # This prevents false positives where different season types collapse to the same game_id
            game_id_col = 'original_game_id' if 'original_game_id' in df.columns else self.config.new_game_id_column

            # Group by game_id and count home teams (is_home_team == 1) and visitor teams (is_home_team == 0)
            game_team_counts = df.groupby(game_id_col)[self.config.home_team_column].agg(['sum', 'count'])
            game_team_counts.columns = ['home_count', 'team_count']

            # Valid games have exactly 1 home team (sum == 1) and 2 total teams (count == 2)
            # This means: 1 home team and 1 visitor team
            valid_games = game_team_counts[(game_team_counts['home_count'] == 1) &
                                          (game_team_counts['team_count'] == 2)].index
            invalid_games = game_team_counts[~((game_team_counts['home_count'] == 1) &
                                               (game_team_counts['team_count'] == 2))].index

            # Split into valid and invalid dataframes
            valid_df = df[df[game_id_col].isin(valid_games)].copy()
            invalid_df = df[df[game_id_col].isin(invalid_games)].copy()

            # Drop original_game_id from valid dataframe as it's no longer needed
            if 'original_game_id' in valid_df.columns:
                valid_df = valid_df.drop(columns=['original_game_id'])

            self.app_logger.structured_log(logging.INFO, "Home/visitor validation complete",
                           total_games=len(game_team_counts),
                           valid_games=len(valid_games),
                           invalid_games=len(invalid_games),
                           valid_records=len(valid_df),
                           invalid_records=len(invalid_df),
                           validation_column=game_id_col)

            if len(invalid_games) > 0:
                self.app_logger.structured_log(logging.WARNING, "Invalid home/visitor assignments detected",
                               invalid_game_ids=invalid_games.tolist()[:10])  # Only log first 10 to avoid log bloat

            return valid_df, invalid_df

        except Exception as e:
            raise DataProcessingError("Error in validate_home_visitor_teams",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def save_invalid_records(self, invalid_team_df: pd.DataFrame, invalid_game_df: pd.DataFrame = None):
        """
        Save invalid records to files with '-invalid' suffix in the processed data directory.

        Args:
            invalid_team_df (pd.DataFrame): Invalid team-centric dataframe.
            invalid_game_df (pd.DataFrame): Invalid game-centric dataframe (optional, for after merge_team_data).

        Raises:
            DataProcessingError: If there's an error during saving.
        """
        if self.app_file_handler is None:
            self.app_logger.structured_log(logging.WARNING,
                           "Cannot save invalid records: app_file_handler not provided")
            return

        self.app_logger.structured_log(logging.INFO, "Starting save_invalid_records",
                       invalid_team_records=len(invalid_team_df),
                       invalid_game_records=len(invalid_game_df) if invalid_game_df is not None else 0)
        try:
            if len(invalid_team_df) > 0:
                # Get base filename and add '-invalid' suffix
                team_file_base = self.config.team_centric_data_file.replace('.csv', '')
                invalid_team_file = f"{team_file_base}-invalid.csv"

                # Construct full path using processed_data_directory
                invalid_team_path = self.app_file_handler.join_paths(
                    self.config.processed_data_directory,
                    invalid_team_file
                )

                # Ensure directory exists
                self.app_file_handler.ensure_directory(self.config.processed_data_directory)

                # Save invalid team records
                self.app_file_handler.write_csv(invalid_team_df, invalid_team_path)

                self.app_logger.structured_log(logging.INFO, "Invalid team records saved",
                               file_path=invalid_team_path,
                               record_count=len(invalid_team_df))

            if invalid_game_df is not None and len(invalid_game_df) > 0:
                # Get base filename and add '-invalid' suffix
                game_file_base = self.config.game_centric_data_file.replace('.csv', '')
                invalid_game_file = f"{game_file_base}-invalid.csv"

                # Construct full path using processed_data_directory
                invalid_game_path = self.app_file_handler.join_paths(
                    self.config.processed_data_directory,
                    invalid_game_file
                )

                # Save invalid game records
                self.app_file_handler.write_csv(invalid_game_df, invalid_game_path)

                self.app_logger.structured_log(logging.INFO, "Invalid game records saved",
                               file_path=invalid_game_path,
                               record_count=len(invalid_game_df))

        except Exception as e:
            raise DataProcessingError("Error in save_invalid_records",
                                      error_message=str(e),
                                      invalid_team_records=len(invalid_team_df),
                                      invalid_game_records=len(invalid_game_df) if invalid_game_df is not None else 0)
        


    @log_performance
    def _merge_dataframes(self, scraped_dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all the different stat type dataframes into a single dataframe.

        Args:
            scraped_dataframes (List[pd.DataFrame]): List of dataframes to merge.

        Returns:
            pd.DataFrame: Merged dataframe.

        Raises:
            DataProcessingError: If there's an error during the merging process.
        """
        self.app_logger.structured_log(logging.INFO, "Starting merge_dataframes",
                       dataframe_count=len(scraped_dataframes))
        try:

            merged = None
            for i, df in enumerate(scraped_dataframes):
                df.columns = df.columns.str.replace('\xa0', ' ') # Replace non-breaking space with regular space
                df = df.rename(columns={'TEAM': 'Team', 'MATCH UP': 'Match Up', 'GAME DATE': 'Game Date'})
                if i == 0:
                    merged = df
                else:
                    merged = pd.merge(merged, df, on=[self.config.game_id_column, self.config.team_id_column], suffixes=('', '_dupe'))

            merged = merged.drop(columns=merged.filter(regex='_dupe').columns)
            self.app_logger.structured_log(logging.INFO, "Dataframes merged successfully",
                           merged_shape=merged.shape)    
                                                 
            return merged
        
        except Exception as e:
            raise DataProcessingError("Error in merge_dataframes",
                                      error_message=str(e),
                                      dataframe_count=len(scraped_dataframes))

    @log_performance
    def _handle_anomalous_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle occurrences of anomalous data. The data from the NBA website is very clean, but there are still some anomalies.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe with missing data handled.

        Raises:
            DataProcessingError: If there's an error during missing data handling.
        """
        self.app_logger.structured_log(logging.INFO, "Starting handle_anomalous_data",
                       initial_nan_count=int(df.isna().sum().sum()))  
        try:
            # Drop rows where 'W/L' is NaN. These games were not played. Normally these are not included in the data, but there was a case where this happened.
            df = df.dropna(subset=['W/L'])

            # A team once had no free throw attempts, and the free throw percentage was set as "-" (for 0/0) in a column that is normally numeric.
            # The following code makes sure that any time something like this happens, the free throw percentage is set to a value designated in the config.
            # This code also handles the also unlikely cases if no field goals were attempted or if no 3 pointers were attempted.
            for col in df.columns:
                if '%' in col:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(self.config.default_percentage_value)

            # check that all the nans are gone
            remaining_nans = int(df.isna().sum().sum())
            self.app_logger.structured_log(logging.INFO, "Anomalous data handling complete",
                           remaining_nan_count=remaining_nans)
            return df
        except Exception as e:
            raise DataProcessingError("Error in handle_anomalous_data",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def _transform_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Transform data by renaming columns, extracting new columns, and reordering columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.

        Raises:
            DataProcessingError: If there's an error during data transformation.
        """
        self.app_logger.structured_log(logging.INFO, "Starting transform_data",
                       initial_column_count=len(df.columns))
        try:
            df, column_mapping = self._rename_columns(df)
            df = self._extract_new_columns(df)
            df = self._reorder_columns(df)

            
            self.app_logger.structured_log(logging.INFO, "Data transformation complete",
                           final_column_count=len(df.columns))
            return df, column_mapping
        except Exception as e:
            raise DataProcessingError("Error in transform_data",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)



    @log_performance
    def _rename_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Rename columns of the dataframe for consistency and to remove special characters and spaces.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: Processed dataframe and column mapping dictionary.

        Raises:
            DataProcessingError: If there's an error during column renaming.
        """
        self.app_logger.structured_log(logging.INFO, "Starting rename_columns",
                       original_column_count=len(df.columns))
        try:
            original_columns = df.columns.tolist()

            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df.columns = df.columns.str.replace('%', '_pct_')
            df.columns = df.columns.str.replace('rtg', '_rtg')  
            df.columns = df.columns.str.replace('__', '_')
            df.columns = df.columns.str.rstrip('_')
            df.columns = df.columns.str.lstrip('_')
            df.columns = df.columns.str.replace('+/-', 'plus_minus')
            df.columns = df.columns.str.replace('ast/to', 'ast_turnover_ratio')

            column_mapping = dict(zip(original_columns, df.columns.tolist()))
            self.app_logger.structured_log(logging.INFO, "Columns renamed successfully",
                           renamed_column_count=len(column_mapping))
            return df, column_mapping
        except Exception as e:
            raise DataProcessingError("Error in rename_columns",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def _extract_new_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract new columns from existing data, such as flagging the home team, overtime games, and playoffs.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with new columns added.

        Raises:
            DataProcessingError: If there's an error during new column extraction.
        """
        initial_column_count=len(df.columns)
        self.app_logger.structured_log(logging.INFO, "Starting extract_new_columns",
                       initial_column_count=initial_column_count)
        try:
            # note that a previous step converted column names to lowercase and substituted an underscore for a space
            df[self.config.win_column] = df["w/l"].str.contains("W").astype(int)
            df = df.drop(columns=['w/l'])

            df[self.config.home_team_column] = df["match_up"].str.contains("vs.").astype(int)
            df[self.config.is_overtime_column] = (df["min"] > self.config.regular_game_minutes).astype(int)

            # Store original game_id before modification for validation purposes
            df[self.config.new_game_id_column] = df[self.config.new_game_id_column].astype(str)
            df['original_game_id'] = df[self.config.new_game_id_column]

            df[self.config.season_column] = df[self.config.new_game_id_column].str[1:3].astype(int) + self.config.season_year_offset
            df[self.config.sub_season_id_column] = df[self.config.new_game_id_column].str[:1].astype(int)
            df[self.config.new_game_id_column] = df[self.config.new_game_id_column].str[1:]
            df[self.config.is_playoff_column] = (df[self.config.sub_season_id_column] > self.config.regular_season_game_id_threshold).astype(int)
            
            

            new_column_count = len(df.columns) - initial_column_count
            self.app_logger.structured_log(logging.INFO, "New columns extracted successfully",
                           new_column_count=new_column_count)
            return df
        except Exception as e:
            raise DataProcessingError("Error in extract_new_columns",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns of the dataframe so that the columns are more logically ordered.
         - All game info (game id, date, is playoff, etc ...) is grouped together and first,
         - followed by the team info (team id, team name, is home team, etc...),
         - and then the team stats are grouped together.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with reordered columns.

        Raises:
            DataProcessingError: If there's an error during column reordering.
        """
        self.app_logger.structured_log(logging.INFO, "Starting reorder_columns",
                       initial_column_count=len(df.columns))
        try:
            all_columns = df.columns.tolist()

            game_info = self.config.game_info_columns
            team_info = self.config.team_info_columns
            # Exclude original_game_id from the reordered dataframe (it's kept temporarily for validation)
            team_stats = [col for col in all_columns if col not in game_info + team_info and col != 'original_game_id']

            reordered_df = df[game_info + team_info + team_stats]

            # Keep original_game_id at the end for validation (will be dropped later)
            if 'original_game_id' in all_columns:
                reordered_df['original_game_id'] = df['original_game_id']

            self.app_logger.structured_log(logging.INFO, "Columns reordered successfully",
                           reordered_column_count=len(reordered_df.columns))
            return reordered_df
        except Exception as e:
            raise DataProcessingError("Error in reorder_columns",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)



