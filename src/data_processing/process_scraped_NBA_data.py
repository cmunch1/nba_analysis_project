import pandas as pd
import logging
from typing import List, Tuple, Dict
from ..config.config import AbstractConfig

logger = logging.getLogger(__name__)

class ProcessScrapedNBAData:
    """
    A class for processing scraped NBA game data.

    This class provides methods to clean, transform, and prepare the scraped data
    for further analysis or model training.
    """

    def __init__(self, config: AbstractConfig):
        """
        Initialize the ProcessScrapedData class.

        Args:
            config (AbstractConfig): Configuration object containing processing parameters.
        """
        self.config = config
        logger.info("ProcessScrapedData initialized")

    def merge_scraped_dataframes(self, scraped_dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple scraped dataframes into a single dataframe.

        Args:
            scraped_dataframes (List[pd.DataFrame]): List of dataframes to merge.

        Returns:
            pd.DataFrame: Merged dataframe.
        """
        logger.info("Merging scraped dataframes")
        try:
            for df in scraped_dataframes:
                if 'TEAM' in df.columns:
                    df = df.rename(columns={'TEAM': 'Team', 'MATCH UP': 'Match Up', 'GAME DATE': 'Game Date'})
                df.columns = df.columns.str.replace('\xa0', ' ')

            merged = None
            for i, df in enumerate(scraped_dataframes):
                if i == 0:
                    merged = df
                else:
                    merged = pd.merge(merged, df, on=['GAME_ID', 'TEAM_ID'], suffixes=('', '_dupe'))

            merged = merged.drop(columns=merged.filter(regex='_dupe').columns)
            logger.info(f"Merged dataframe shape: {merged.shape}")
            return merged
        except Exception as e:
            logger.error(f"Error in merge_scraped_dataframes: {str(e)}")
            raise

    def process_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process NaN values in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe with NaNs handled.
        """
        logger.info("Processing NaN values")
        try:
            df = df.dropna(subset=['W/L'])

            for col in df.columns:
                if '%' in col:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(self.config.default_percentage_value)

            logger.info(f"NaN processing complete. Remaining NaNs: {df.isna().sum().sum()}")
            return df
        except Exception as e:
            logger.error(f"Error in process_nans: {str(e)}")
            raise

    def rename_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Rename columns of the dataframe for consistency.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: Processed dataframe and column mapping dictionary.
        """
        logger.info("Renaming columns")
        try:
            original_columns = df.columns.tolist()

            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df.columns = df.columns.str.replace('%', 'pct_')
            df.columns = df.columns.str.replace('_$', '')

            column_mapping = dict(zip(original_columns, df.columns.tolist()))
            logger.info(f"Renamed {len(column_mapping)} columns")
            return df, column_mapping
        except Exception as e:
            logger.error(f"Error in rename_columns: {str(e)}")
            raise

    def extract_new_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract new columns from existing data.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with new columns added.
        """
        logger.info("Extracting new columns")
        try:
            df["is_win"] = df["W/L"].str.contains("W").astype(int)
            df = df.drop(columns=['W/L'])

            df["is_home_team"] = df["Match Up"].str.contains("vs.").astype(int)
            df["is_overtime"] = (df["MIN"] > self.config.regular_game_minutes).astype(int)

            df["GAME_ID"] = df["GAME_ID"].astype(str)
            df["season"] = df["GAME_ID"].str[1:3].astype(int) + self.config.season_year_offset
            df["is_playoff"] = (df["GAME_ID"].str[0].astype(int) > self.config.regular_season_game_id_threshold).astype(int)

            logger.info(f"Extracted {5} new columns")
            return df
        except Exception as e:
            logger.error(f"Error in extract_new_columns: {str(e)}")
            raise

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns of the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with reordered columns.
        """
        logger.info("Reordering columns")
        try:
            all_columns = df.columns.tolist()

            game_info = self.config.game_info_columns
            team_info = self.config.team_info_columns
            team_stats = [col for col in all_columns if col not in game_info + team_info]

            reordered_df = df[game_info + team_info + team_stats]
            logger.info(f"Reordered {len(reordered_df.columns)} columns")
            return reordered_df
        except Exception as e:
            logger.error(f"Error in reorder_columns: {str(e)}")
            raise