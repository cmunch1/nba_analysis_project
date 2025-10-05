"""
base_scraper_classes.py

This module defines base classes for various scraper components used in the NBA data scraping application.
These base classes provide a common interface for different scraper implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from selenium.webdriver.remote.webelement import WebElement
import pandas as pd

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.framework.data_access.base_data_access import BaseDataAccess

class BaseNbaScraper(ABC):
    @abstractmethod
    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        pass

    @abstractmethod
    def scrape_and_save_matchups_for_day(self, search_day: str) -> None:
        pass

class BaseBoxscoreScraper(ABC):
    @abstractmethod
    def __init__(self, config: BaseConfigManager, data_access: BaseDataAccess):
        pass

    @abstractmethod
    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        """Scrape and save boxscores for all stat types and specified seasons."""
        pass

    @abstractmethod
    def scrape_stat_type(self, seasons: List[str], first_start_date: str, stat_type: str) -> pd.DataFrame:
        """Scrape a specific stat type for multiple seasons."""
        pass

    @abstractmethod
    def scrape_sub_seasons(self, season: str, start_date: str, end_date: str, stat_type: str) -> pd.DataFrame:
        """Scrape data for all sub-seasons within a given season."""
        pass

class BaseScheduleScraper(ABC):
    @abstractmethod
    def __init__(self, config: BaseConfigManager, data_access: BaseDataAccess):
        pass

    @abstractmethod
    def scrape_and_save_matchups_for_day(self, search_day: str) -> None:
        """Scrape and save matchups for a specific day."""
        pass

class BaseWebDriver(ABC):
    @abstractmethod
    def __init__(self, config: BaseConfigManager):
        pass

class BasePageScraper(ABC):
    @abstractmethod
    def __init__(self, config: BaseConfigManager, web_driver: BaseWebDriver):
        pass
    
    @abstractmethod
    def go_to_url(self, url: str) -> bool:
        """Navigate to the specified URL."""
        pass

    @abstractmethod
    def get_elements_by_class(self, class_name: str, parent_element: Optional[WebElement] = None) -> Optional[List[WebElement]]:
        """Retrieve elements by class name, optionally within a parent element."""
        pass

    @abstractmethod
    def get_links_by_class(self, class_name: str, parent_element: Optional[WebElement] = None) -> Optional[List[WebElement]]:
        """Retrieve link elements (a tags) by class name, optionally within a parent element."""
        pass

    @abstractmethod
    def scrape_page_table(self, url: str, table_class: str, pagination_class: str, dropdown_class: str) -> Optional[WebElement]:
        """Scrape a table from a web page, handling pagination if necessary."""
        pass





