from abc import ABC, abstractmethod
from ..config.abstract_config import AbstractConfig
from ..data_access.abstract_data_access import AbstractDataAccess
from typing import List
import pandas as pd

class AbstractDataValidator(ABC):
    @abstractmethod
    def __init__(self, config: AbstractConfig):
        pass

    @abstractmethod
    def validate_scraped_dataframes(self, scraped_dataframes: List[pd.DataFrame], file_names: List[str]) -> bool:
        pass

    @abstractmethod
    def validate_processed_dataframe(self, df: pd.DataFrame, file_name: str) -> bool:
        pass


