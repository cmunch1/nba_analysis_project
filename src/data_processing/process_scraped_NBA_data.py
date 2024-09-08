import pandas as pd
import logging
from typing import List, Tuple, Dict
from ..config.config import AbstractConfig
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import DataProcessingError
from .abstract_data_processing_classes import AbstractNBADataProcessor

logger = logging.getLogger(__name__)

class ProcessScrapedNBAData(AbstractNBADataProcessor):
    @log_performance
    def __init__(self,  config: AbstractConfig):
        """
        Initialize the ProcessScrapedNBAData class.

        Args:
            config (AbstractConfig): Configuration object containing processing parameters.
        """
        self.config = config
        
        structured_log(logger, logging.INFO, "ProcessScrapedNBAData initialized",
                       config_type=type(config).__name__)
        
    @log_performance
    def process_data(self, scraped_dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Process the scraped NBA data through all steps.

        Args:
            scraped_dataframes (List[pd.DataFrame]): List of scraped dataframes.

        Returns:
            pd.DataFrame: Fully processed dataframe.

        Raises:
            DataProcessingError: If there's an error during any step of the data processing.
        """
        
        structured_log(logger, logging.INFO, "Starting process_data",
                       dataframe_count=len(scraped_dataframes))
        try:
                        
            df = self._merge_dataframes(scraped_dataframes)
            df = self._handle_anomalous_data(df)
            df = self._transform_data(df)
                      
            structured_log(logger, logging.INFO, "Data processing completed successfully",
                           final_dataframe_shape=df.shape)
            return df
        
        except (DataProcessingError) as e:
            structured_log(logger, logging.ERROR, "Error in process_data",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Unexpected error in process_data",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise DataProcessingError("Unexpected error in process_data",
                                      error_message=str(e))


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
        structured_log(logger, logging.INFO, "Starting merge_dataframes",
                       dataframe_count=len(scraped_dataframes))
        try:
            for df in scraped_dataframes:
                if 'TEAM' in df.columns:
                    df = df.rename(columns={'TEAM': 'Team', 'MATCH UP': 'Match Up', 'GAME DATE': 'Game Date'})
                df.columns = df.columns.str.replace('\xa0', ' ') # Replace non-breaking space with regular space

            merged = None
            for i, df in enumerate(scraped_dataframes):
                if i == 0:
                    merged = df
                else:
                    merged = pd.merge(merged, df, on=['GAME_ID', 'TEAM_ID'], suffixes=('', '_dupe'))

            merged = merged.drop(columns=merged.filter(regex='_dupe').columns)
            structured_log(logger, logging.INFO, "Dataframes merged successfully",
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
        structured_log(logger, logging.INFO, "Starting handle_anomalous_data",
                       initial_nan_count=df.isna().sum().sum())
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
            remaining_nans = df.isna().sum().sum()
            structured_log(logger, logging.INFO, "Anomalous data handling complete",
                           remaining_nan_count=remaining_nans)
            return df
        except Exception as e:
            raise DataProcessingError("Error in handle_anomalous_data",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by renaming columns, extracting new columns, and reordering columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.

        Raises:
            DataProcessingError: If there's an error during data transformation.
        """
        structured_log(logger, logging.INFO, "Starting transform_data",
                       initial_column_count=len(df.columns))
        try:
            df, _ = self._rename_columns(df)
            df = self._extract_new_columns(df)
            df = self._reorder_columns(df)
            
            structured_log(logger, logging.INFO, "Data transformation complete",
                           final_column_count=len(df.columns))
            return df
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
        structured_log(logger, logging.INFO, "Starting rename_columns",
                       original_column_count=len(df.columns))
        try:
            original_columns = df.columns.tolist()

            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df.columns = df.columns.str.replace('%', 'pct_')
            df.columns = df.columns.str.replace('_$', '')

            column_mapping = dict(zip(original_columns, df.columns.tolist()))
            structured_log(logger, logging.INFO, "Columns renamed successfully",
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
        structured_log(logger, logging.INFO, "Starting extract_new_columns",
                       initial_column_count=len(df.columns))
        try:
            df["is_win"] = df["W/L"].str.contains("W").astype(int)
            df = df.drop(columns=['W/L'])

            df["is_home_team"] = df["Match Up"].str.contains("vs.").astype(int)
            df["is_overtime"] = (df["MIN"] > self.config.regular_game_minutes).astype(int)

            df["GAME_ID"] = df["GAME_ID"].astype(str)
            df["season"] = df["GAME_ID"].str[1:3].astype(int) + self.config.season_year_offset
            df["is_playoff"] = (df["GAME_ID"].str[0].astype(int) > self.config.regular_season_game_id_threshold).astype(int)

            new_column_count = len(df.columns) - initial_column_count
            structured_log(logger, logging.INFO, "New columns extracted successfully",
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
        structured_log(logger, logging.INFO, "Starting reorder_columns",
                       initial_column_count=len(df.columns))
        try:
            all_columns = df.columns.tolist()

            game_info = self.config.game_info_columns
            team_info = self.config.team_info_columns
            team_stats = [col for col in all_columns if col not in game_info + team_info]

            reordered_df = df[game_info + team_info + team_stats]
            structured_log(logger, logging.INFO, "Columns reordered successfully",
                           reordered_column_count=len(reordered_df.columns))
            return reordered_df
        except Exception as e:
            raise DataProcessingError("Error in reorder_columns",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)



