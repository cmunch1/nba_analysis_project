import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import Mock, patch
from selenium.webdriver.remote.webelement import WebElement

from nba_app.webscraping.boxscore_scraper import BoxscoreScraper
from ml_framework.core.error_handling.error_handler import (
    DataValidationError, ConfigurationError, DataExtractionError, ScrapingError
)

@pytest.fixture
def mock_config():
    config = Mock()
    config.stat_types = ['traditional', 'advanced']
    config.regular_season_text = "Regular+Season"
    config.playoffs_season_text = "Playoffs"
    config.play_in_season_text = "Play+In"
    config.play_in_month = 4
    config.regular_season_start_month = 10
    config.off_season_start_month = 6
    config.table_class_name = "table-class"
    config.pagination_class_name = "pagination-class"
    config.dropdown_class_name = "dropdown-class"
    config.teams_and_games_class_name = "links-class"
    config.team_id_column = "TEAM_ID"
    config.game_id_column = "GAME_ID"
    config.nba_boxscores_url = "https://stats.nba.com/stats/leaguegamelog"
    return config

@pytest.fixture
def mock_data_access():
    return Mock()

@pytest.fixture
def mock_page_scraper():
    return Mock()

@pytest.fixture
def scraper(mock_config, mock_data_access, mock_page_scraper):
    return BoxscoreScraper(mock_config, mock_data_access, mock_page_scraper)

def test_initialization(mock_config, mock_data_access, mock_page_scraper):
    scraper = BoxscoreScraper(mock_config, mock_data_access, mock_page_scraper)
    assert scraper.config == mock_config
    assert scraper.data_access == mock_data_access
    assert scraper.page_scraper == mock_page_scraper

def test_initialization_error():
    with pytest.raises(ConfigurationError):
        BoxscoreScraper(None, None, None)

def test_is_valid_date():
    assert BoxscoreScraper._is_valid_date("12/25/2023") == True
    assert BoxscoreScraper._is_valid_date("13/25/2023") == False
    assert BoxscoreScraper._is_valid_date("invalid") == False

def test_determine_sub_season_types(scraper):
    # Test regular season only
    types = scraper._determine_sub_season_types(
        "2023", "10/01/2023", "03/31/2024"
    )
    assert types == [scraper.config.regular_season_text]

    # Test playoffs only
    types = scraper._determine_sub_season_types(
        "2023", "04/15/2024", "06/01/2024"
    )
    assert types == [scraper.config.playoffs_season_text]

    # Test full season (including play-in)
    types = scraper._determine_sub_season_types(
        "2023", "10/01/2023", "06/01/2024"
    )
    assert set(types) == {
        scraper.config.regular_season_text,
        scraper.config.playoffs_season_text,
        scraper.config.play_in_season_text
    }

def test_construct_nba_url(scraper):
    url = scraper._construct_nba_url(
        stat_type="traditional",
        season_type="Regular+Season",
        Season="2023",
        DateFrom="10/01/2023",
        DateTo="04/01/2024"
    )
    assert "traditional" in url
    assert "Regular+Season" in url
    assert "2023" in url
    assert "10/01/2023" in url
    assert "04/01/2024" in url

def test_convert_table_to_df(scraper):
    mock_table = Mock(spec=WebElement)
    mock_table.get_attribute.return_value = """
        <table>
            <tr><th>Column1</th><th>Column2</th></tr>
            <tr><td>Value1</td><td>Value2</td></tr>
        </table>
    """
    
    with patch('pandas.read_html') as mock_read_html:
        mock_df = pd.DataFrame({'Column1': ['Value1'], 'Column2': ['Value2']})
        mock_read_html.return_value = [mock_df]
        
        # Mock the ID extraction method
        with patch.object(scraper, '_extract_team_and_game_ids_boxscores') as mock_extract:
            mock_extract.return_value = (pd.Series(['1']), pd.Series(['100']))
            
            result = scraper._convert_table_to_df(mock_table)
            
            assert isinstance(result, pd.DataFrame)
            assert scraper.config.team_id_column in result.columns
            assert scraper.config.game_id_column in result.columns

def test_scrape_and_save_all_boxscores_validation(scraper):
    with pytest.raises(DataValidationError):
        scraper.scrape_and_save_all_boxscores([], "10/01/2023")
    
    with pytest.raises(DataValidationError):
        scraper.scrape_and_save_all_boxscores(["2023"], "invalid_date")

def test_scrape_stat_type_validation(scraper):
    with pytest.raises(DataValidationError):
        scraper.scrape_stat_type(["2023"], "10/01/2023", "invalid_stat_type")

def test_scrape_boxscores_table(scraper):
    mock_table = Mock(spec=WebElement)
    scraper.page_scraper.scrape_page_table.return_value = mock_table
    
    result = scraper.scrape_boxscores_table(
        Season="2023",
        DateFrom="10/01/2023",
        DateTo="04/01/2024",
        stat_type="traditional",
        season_type="Regular+Season"
    )
    
    assert result == mock_table
    scraper.page_scraper.scrape_page_table.assert_called_once()

def test_scrape_boxscores_table_error(scraper):
    scraper.page_scraper.scrape_page_table.side_effect = ScrapingError("Test error")
    
    with pytest.raises(ScrapingError):
        scraper.scrape_boxscores_table(
            Season="2023",
            DateFrom="10/01/2023",
            DateTo="04/01/2024"
        )