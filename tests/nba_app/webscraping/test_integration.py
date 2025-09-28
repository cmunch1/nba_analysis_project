import pytest
from unittest.mock import Mock, patch, PropertyMock
from selenium.webdriver.remote.webelement import WebElement
import pandas as pd
from datetime import datetime

from nba_app.webscraping.nba_scraper import NbaScraper
from nba_app.webscraping.boxscore_scraper import BoxscoreScraper
from nba_app.webscraping.schedule_scraper import ScheduleScraper
from platform_core.core.error_handling.error_handler import (
    ScrapingError,
    PageLoadError,
    ElementNotFoundError,
    DataExtractionError
)

@pytest.fixture
def mock_config():
    config = Mock()
    # Configure stat types
    type(config).stat_types = PropertyMock(return_value=['traditional', 'advanced'])
    type(config).day_class_name = PropertyMock(return_value='day-class')
    type(config).games_per_day_class_name = PropertyMock(return_value='games-class')
    
    # Configure season dates
    type(config).off_season_start_month = PropertyMock(return_value="08")
    type(config).regular_season_start_month = PropertyMock(return_value="10")
    
    # Configure URLs and class names
    config.nba_schedule_url = "https://example.com/schedule"
    config.nba_boxscores_url = "https://example.com/boxscores"
    config.table_class_name = "table-class"
    
    return config

@pytest.fixture
def setup_scrapers(mock_config):
    """Setup scrapers with mocked dependencies"""
    data_access = Mock()
    page_scraper = Mock()
    
    boxscore_scraper = BoxscoreScraper(mock_config, data_access, page_scraper)
    schedule_scraper = ScheduleScraper(mock_config, data_access, page_scraper)
    nba_scraper = NbaScraper(mock_config, boxscore_scraper, schedule_scraper)
    
    return {
        'nba_scraper': nba_scraper,
        'boxscore_scraper': boxscore_scraper,
        'schedule_scraper': schedule_scraper,
        'page_scraper': page_scraper,
        'data_access': data_access
    }

def test_full_scraping_pipeline(setup_scrapers):
    """Test the complete scraping pipeline from schedule to boxscores"""
    scrapers = setup_scrapers
    
    # Mock schedule scraping
    scrapers['schedule_scraper'].get_game_ids_for_date = Mock(return_value=["100"])
    
    # Mock successful page loading
    scrapers['page_scraper'].go_to_url = Mock(return_value=True)
    scrapers['page_scraper'].wait_for_page_load = Mock(return_value=True)
    
    # Mock table scraping
    mock_table = Mock(spec=WebElement)
    mock_table.get_attribute.return_value = "<table><tr><td>Data</td></tr></table>"
    scrapers['page_scraper'].scrape_page_table = Mock(return_value=mock_table)
    
    # Mock DataFrame operations with consistent lengths
    mock_df = pd.DataFrame({
        'TEAM_ID': ['1', '2'],
        'GAME_ID': ['100', '100'],
        'PTS': [100, 98]
    })
    
    # Mock boxscore operations
    mock_scrape_stat = Mock(return_value=mock_df)
    scrapers['boxscore_scraper'].scrape_stat_type = mock_scrape_stat
    
    with patch('pandas.read_html', return_value=[mock_df]):
        scrapers['nba_scraper'].scrape_and_save_all_boxscores(["2023"], "01/01/2023")
        
        assert scrapers['page_scraper'].go_to_url.called
        assert mock_scrape_stat.called
        assert scrapers['data_access'].save_dataframes.called
        
        saved_data = scrapers['data_access'].save_dataframes.call_args[0][0]
        assert isinstance(saved_data, dict)
        assert 'traditional' in saved_data

def test_error_handling_chain(setup_scrapers):
    """Test error handling and recovery at different stages"""
    scrapers = setup_scrapers
    
    # Test page load failure
    scrapers['page_scraper'].go_to_url.side_effect = PageLoadError("Failed to load")
    with pytest.raises(ScrapingError):
        scrapers['nba_scraper'].scrape_and_save_all_boxscores(["2023"], "01/01/2023")
    
    # Test element not found
    scrapers['page_scraper'].go_to_url.side_effect = None
    scrapers['page_scraper'].get_elements_by_class.side_effect = ElementNotFoundError("Element not found")
    with pytest.raises(ScrapingError):
        scrapers['nba_scraper'].scrape_and_save_matchups_for_day("MON")
    
    # Test data extraction failure - Fixed mock setup
    scrapers['page_scraper'].get_elements_by_class.side_effect = None
    mock_scrape_stat = Mock(side_effect=DataExtractionError("Failed to extract data"))
    scrapers['boxscore_scraper'].scrape_stat_type = mock_scrape_stat
    
    with pytest.raises(ScrapingError):
        scrapers['nba_scraper'].scrape_and_save_all_boxscores(["2023"], "01/01/2023")

def test_schedule_scraping(setup_scrapers):
    """Test schedule scraping functionality"""
    scrapers = setup_scrapers
    
    # Mock successful schedule page elements
    mock_day = Mock()
    mock_day.text = "MON"
    mock_games = Mock()
    
    # Create two mock links for home and away teams
    mock_link_away = Mock()
    mock_link_home = Mock()
    mock_link_away.get_attribute.return_value = "https://example.com/team/1610612743/profile"
    mock_link_home.get_attribute.return_value = "https://example.com/team/1610612744/profile"
    
    # Configure schedule_preview_text in config
    scrapers['schedule_scraper'].config.schedule_preview_text = "PREVIEW"
    
    # Mock game links for game IDs
    mock_game_link = Mock()
    mock_game_link.text = "PREVIEW"
    mock_game_link.get_attribute.return_value = "https://example.com/game/0022301234-00"
    
    # Set up the side effects for multiple get_links_by_class calls
    scrapers['page_scraper'].get_elements_by_class.return_value = [mock_day, mock_games]
    scrapers['page_scraper'].get_links_by_class.side_effect = [
        [mock_link_away, mock_link_home],  # First call for team links
        [mock_game_link]  # Second call for game links
    ]
    
    result = scrapers['nba_scraper'].scrape_and_save_matchups_for_day("MON")
    
    assert result is True
    assert scrapers['data_access'].save_dataframes.called