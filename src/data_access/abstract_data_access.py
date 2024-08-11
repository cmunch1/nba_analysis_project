"""
abstract_data_access.py

This file contains abstract base classes for data access operations.
These classes define the interface for data access operations, which can be implemented
by concrete classes for different data sources (e.g., CSV files, databases, APIs).
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List



class AbstractDataAccess(ABC):
    @abstractmethod
    def save_scraped_data(self, df: pd.DataFrame, file_name: str, cumulative: bool = False) -> None:
        """
        Save the scraped data to a storage medium.

        Args:
            df (pd.DataFrame): The scraped data to save.
            file_name (str): The name of the file to save the data to.
            cumulative (bool): Whether to save to newly scraped data or the cumulative scraped data.
        """
        pass

    @abstractmethod
    def load_scraped_data(self, cumulative: bool = False) -> List[pd.DataFrame]:
        """
        Load the scraped data from a storage medium.

        Args:
            cumulative (bool): Whether to load the newly scraped data or the cumulative scraped data.

        Returns:
            List[pd.DataFrame]: List of DataFrames containing the loaded data.
        """
        pass