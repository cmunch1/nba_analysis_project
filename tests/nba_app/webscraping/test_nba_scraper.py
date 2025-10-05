import pytest
from unittest.mock import Mock, patch

from nba_app.webscraping.nba_scraper import NbaScraper
from nba_app.webscraping.base_scraper_classes import (
    BaseBoxscoreScraper,
    BaseScheduleScraper
)
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.error_handling.error_handler import (
    ConfigurationError,
    DataValidationError,
    ScrapingError,
    DataStorageError
)

class MockBoxscoreScraper(BaseBoxscoreScraper):
    def scrape_and_save_all_boxscores(self, seasons, first_start_date):
        pass

class MockScheduleScraper(BaseScheduleScraper):
    def scrape_and_save_matchups_for_day(self, search_day):
        pass

class MockConfig(BaseConfigManager):
    pass

@pytest.fixture
def mock_config():
    return Mock(spec=MockConfig)

@pytest.fixture
def mock_boxscore_scraper():
    return Mock(spec=MockBoxscoreScraper)

@pytest.fixture
def mock_schedule_scraper():
    return Mock(spec=MockScheduleScraper)

@pytest.fixture
def scraper(mock_config, mock_boxscore_scraper, mock_schedule_scraper):
    return NbaScraper(mock_config, mock_boxscore_scraper, mock_schedule_scraper)

def test_initialization(mock_config, mock_boxscore_scraper, mock_schedule_scraper):
    scraper = NbaScraper(mock_config, mock_boxscore_scraper, mock_schedule_scraper)
    assert scraper._config == mock_config
    assert scraper._boxscore_scraper == mock_boxscore_scraper
    assert scraper._schedule_scraper == mock_schedule_scraper

def test_initialization_invalid_scrapers():
    with pytest.raises(ConfigurationError):
        NbaScraper(Mock(), Mock(), Mock())  # Invalid scraper types

def test_scrape_and_save_all_boxscores_validation(scraper):
    with pytest.raises(DataValidationError):
        scraper.scrape_and_save_all_boxscores([], "10/01/2023")
    
    with pytest.raises(DataValidationError):
        scraper.scrape_and_save_all_boxscores(["2023"], "invalid_date")

def test_scrape_and_save_all_boxscores_success(scraper):
    seasons = ["2022-23"]
    start_date = "10/01/2022"
    
    scraper.scrape_and_save_all_boxscores(seasons, start_date)
    
    scraper._boxscore_scraper.scrape_and_save_all_boxscores.assert_called_once_with(
        seasons, start_date
    )

def test_scrape_and_save_all_boxscores_scraping_error(scraper):
    scraper._boxscore_scraper.scrape_and_save_all_boxscores.side_effect = ScrapingError("Test error")
    
    with pytest.raises(ScrapingError):
        scraper.scrape_and_save_all_boxscores(["2022-23"], "10/01/2022")

def test_scrape_and_save_matchups_for_day_validation(scraper):
    with pytest.raises(DataValidationError):
        scraper.scrape_and_save_matchups_for_day("MONDAY")  # Invalid format
    
    with pytest.raises(DataValidationError):
        scraper.scrape_and_save_matchups_for_day("M")  # Too short

def test_scrape_and_save_matchups_for_day_success(scraper):
    search_day = "MON"
    scraper._schedule_scraper.scrape_and_save_matchups_for_day.return_value = True
    
    result = scraper.scrape_and_save_matchups_for_day(search_day)
    
    assert result is True
    scraper._schedule_scraper.scrape_and_save_matchups_for_day.assert_called_once_with(search_day)

def test_scrape_and_save_matchups_for_day_no_matchups(scraper):
    search_day = "MON"
    scraper._schedule_scraper.scrape_and_save_matchups_for_day.return_value = False
    
    result = scraper.scrape_and_save_matchups_for_day(search_day)
    
    assert result is False
    scraper._schedule_scraper.scrape_and_save_matchups_for_day.assert_called_once_with(search_day)

def test_scrape_and_save_matchups_for_day_scraping_error(scraper):
    scraper._schedule_scraper.scrape_and_save_matchups_for_day.side_effect = ScrapingError("Test error")
    
    with pytest.raises(ScrapingError):
        scraper.scrape_and_save_matchups_for_day("MON")

def test_validate_boxscore_input(scraper):
    # Valid input
    scraper._validate_boxscore_input(["2022-23"], "10/01/2022")
    
    # Test empty seasons list
    with pytest.raises(DataValidationError, match="Seasons list cannot be empty"):
        scraper._validate_boxscore_input([], "10/01/2022")
    
    # Test invalid date format
    with pytest.raises(DataValidationError, match="Invalid first_start_date format"):
        scraper._validate_boxscore_input(["2022-23"], "invalid_date")  # Clearly invalid format
        
    with pytest.raises(DataValidationError, match="Invalid first_start_date format"):
        scraper._validate_boxscore_input(["2022-23"], "2022-10-01")  # Wrong separator

def test_validate_search_day(scraper):
    # Valid input
    scraper._validate_search_day("MON")
    
    # Invalid inputs
    with pytest.raises(DataValidationError):
        scraper._validate_search_day("MONDAY")
    
    with pytest.raises(DataValidationError):
        scraper._validate_search_day("M")
