import pytest
from unittest.mock import Mock, patch
from selenium.webdriver.remote.webelement import WebElement
import pandas as pd

from src.webscraping.schedule_scraper import ScheduleScraper
from src.error_handling.custom_exceptions import (
    ScrapingError, DataExtractionError, DataValidationError, ElementNotFoundError,
    PageLoadError, DataStorageError
)

@pytest.fixture
def mock_config():
    config = Mock()
    config.nba_schedule_url = "https://www.nba.com/schedule"
    config.day_class_name = "day-class"
    config.games_per_day_class_name = "games-class"
    config.teams_links_class_name = "teams-links"
    config.game_links_class_name = "game-links"
    config.schedule_preview_text = "PREVIEW"
    config.schedule_visitor_team_id_column = "VISITOR_TEAM_ID"
    config.schedule_home_team_id_column = "HOME_TEAM_ID"
    config.schedule_game_id_column = "GAME_ID"
    config.todays_matchups_file = "matchups.csv"
    config.todays_games_ids_file = "games.csv"
    return config

@pytest.fixture
def mock_data_access():
    return Mock()

@pytest.fixture
def mock_page_scraper():
    return Mock()

@pytest.fixture
def scraper(mock_config, mock_data_access, mock_page_scraper):
    return ScheduleScraper(mock_config, mock_data_access, mock_page_scraper)

def test_initialization(mock_config, mock_data_access, mock_page_scraper):
    scraper = ScheduleScraper(mock_config, mock_data_access, mock_page_scraper)
    assert scraper.config == mock_config
    assert scraper.data_access == mock_data_access
    assert scraper.page_scraper == mock_page_scraper

def test_find_games_for_day_success(scraper):
    mock_day = Mock()
    mock_day.text = "MON 12/25"
    mock_games = Mock()
    
    scraper.page_scraper.get_elements_by_class.side_effect = [[mock_day], [mock_games]]
    
    result = scraper._find_games_for_day("MON")
    assert result == mock_games

def test_find_games_for_day_not_found(scraper):
    mock_day = Mock()
    mock_day.text = "TUE 12/26"
    mock_games = Mock()
    
    scraper.page_scraper.get_elements_by_class.side_effect = [[mock_day], [mock_games]]
    
    result = scraper._find_games_for_day("MON")
    assert result is None

def test_find_games_for_day_element_not_found(scraper):
    scraper.page_scraper.get_elements_by_class.return_value = []
    
    with pytest.raises(ElementNotFoundError):
        scraper._find_games_for_day("MON")

def test_extract_team_ids_schedule_success(scraper):
    mock_games = Mock()
    mock_link1 = Mock()
    mock_link2 = Mock()
    mock_link1.get_attribute.return_value = "https://nba.com/team/1234/profile"
    mock_link2.get_attribute.return_value = "https://nba.com/team/5678/profile"
    
    scraper.page_scraper.get_links_by_class.return_value = [mock_link1, mock_link2]
    
    result = scraper._extract_team_ids_schedule(mock_games)
    assert result == [["1234", "5678"]]

def test_extract_team_ids_schedule_no_links(scraper):
    mock_games = Mock()
    scraper.page_scraper.get_links_by_class.return_value = []
    
    with pytest.raises(ElementNotFoundError):
        scraper._extract_team_ids_schedule(mock_games)

def test_extract_game_ids_schedule_success(scraper):
    mock_games = Mock()
    mock_link = Mock()
    mock_link.text = "PREVIEW"
    mock_link.get_attribute.return_value = "https://nba.com/game/123-00456"
    
    scraper.page_scraper.get_links_by_class.return_value = [mock_link]
    
    result = scraper._extract_game_ids_schedule(mock_games)
    assert result == ["456"]

def test_extract_game_ids_schedule_no_links(scraper):
    mock_games = Mock()
    scraper.page_scraper.get_links_by_class.return_value = []
    
    with pytest.raises(ElementNotFoundError):
        scraper._extract_game_ids_schedule(mock_games)

def test_save_matchups_and_games_success(scraper):
    matchups = [["1234", "5678"]]
    games = ["456"]
    
    scraper._save_matchups_and_games(matchups, games)
    
    scraper.data_access.save_dataframes.assert_called_once()

def test_save_matchups_and_games_validation_error(scraper):
    with pytest.raises(DataValidationError):
        scraper._save_matchups_and_games([], [])

def test_scrape_and_save_matchups_for_day_success(scraper):
    mock_games = Mock()
    with patch.object(scraper, '_find_games_for_day') as mock_find_games, \
         patch.object(scraper, '_extract_team_ids_schedule') as mock_extract_teams, \
         patch.object(scraper, '_extract_game_ids_schedule') as mock_extract_games:
        
        mock_find_games.return_value = mock_games
        mock_extract_teams.return_value = [["1234", "5678"]]
        mock_extract_games.return_value = ["456"]
        
        result = scraper.scrape_and_save_matchups_for_day("MON")
        assert result is True

def test_scrape_and_save_matchups_for_day_no_games(scraper):
    with patch.object(scraper, '_find_games_for_day') as mock_find_games:
        mock_find_games.return_value = None
        
        result = scraper.scrape_and_save_matchups_for_day("MON")
        assert result is False

def test_scrape_and_save_matchups_for_day_page_load_error(scraper):
    scraper.page_scraper.go_to_url.side_effect = PageLoadError("Failed to load page")
    
    with pytest.raises(PageLoadError):
        scraper.scrape_and_save_matchups_for_day("MON")