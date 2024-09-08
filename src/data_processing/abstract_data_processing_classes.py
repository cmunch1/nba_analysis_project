from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any



class AbstractNBADataProcessor(ABC):
    @abstractmethod
    def process_data(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Abstract method to process NBA data.
        
        Args:
            data (List[pd.DataFrame]): The data to process.
        
        Returns:
            pd.DataFrame: Processed data as a single DataFrame.
        """
        pass

    @abstractmethod
    def merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Abstract method to merge multiple DataFrames.
        
        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames to merge.
        
        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        pass

    @abstractmethod
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing data.
        
        Args:
            df (pd.DataFrame): DataFrame with missing data.
        
        Returns:
            pd.DataFrame: DataFrame with missing data handled.
        """
        pass

    @abstractmethod
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to transform data (e.g., create new columns, rename columns).
        
        Args:
            df (pd.DataFrame): DataFrame to transform.
        
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        pass

