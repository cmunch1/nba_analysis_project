from abc import ABC, abstractmethod
from ..config.abstract_config import AbstractConfig
from ..data_access.abstract_data_access import AbstractDataAccess
from typing import List
import pandas as pd

class AbstractDataValidator(ABC):
    @abstractmethod
    def __init__(self, config: AbstractConfig, data_access: AbstractDataAccess):
        pass

    @abstractmethod
    def validate_scraped_dataframes(self, scraped_dataframes: List[pd.DataFrame]) -> bool:
        pass