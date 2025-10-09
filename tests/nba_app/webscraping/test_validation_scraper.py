"""
Tests for the validation_scraper module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement

from nba_app.webscraping.validation_scraper import ValidationScraper
from ml_framework.core.error_handling.error_handler import (
    ConfigurationError,
    ScrapingError,
    DataStorageError
)


@pytest.fixture
def mock_config():
    """Create a mock config for validation scraper."""
    from types import SimpleNamespace

    config = Mock()
    config.validation_data_file = "validation_data.csv"
    config.scoreboard_base_url = "https://www.basketball-reference.com/boxscores"

    # Team mappings - use SimpleNamespace instead of Mock
    team_id_to_abbrev_dict = {
        '1610612747': 'LAL',
        '1610612738': 'BOS',
        '1610612751': 'BKN'
    }
    config.team_id_to_abbrev = SimpleNamespace(**team_id_to_abbrev_dict)

    historical_teams_dict = {
        'SEA': '1610612754',  # Seattle SuperSonics
        'NJN': '1610612751'   # New Jersey Nets
    }
    config.historical_teams = SimpleNamespace(**historical_teams_dict)

    # Selectors
    config.selectors = SimpleNamespace(
        box_score_link='a[href*="/boxscores/"]',
        team_links='a[href*="/teams/"]'
    )

    # Team abbreviation parsing
    config.team_abbrev = SimpleNamespace(url_path_index=2)

    # Parsing config
    config.parsing = SimpleNamespace(verify_home_team_from_url=True)

    # Error handling config
    config.error_handling = SimpleNamespace(
        fail_on_verification_mismatch=False,
        warn_on_unknown_teams=True
    )

    # Rate limiting - make it a dict-like object
    config.rate_limiting = {'delay_between_dates': 0.1}

    return config


@pytest.fixture
def mock_data_access():
    """Create a mock data access object."""
    return Mock()


@pytest.fixture
def mock_page_scraper():
    """Create a mock page scraper."""
    scraper = Mock()
    scraper.page_scraper = Mock()
    scraper.page_scraper.driver = Mock()
    return scraper


@pytest.fixture
def validation_scraper(mock_config, mock_data_access, mock_page_scraper, mock_app_logger, mock_error_handler):
    """Create a validation scraper instance."""
    return ValidationScraper(mock_config, mock_data_access, mock_page_scraper, mock_app_logger, mock_error_handler)


def test_initialization(mock_config, mock_data_access, mock_page_scraper, mock_app_logger, mock_error_handler):
    """Test that ValidationScraper initializes correctly."""
    scraper = ValidationScraper(mock_config, mock_data_access, mock_page_scraper, mock_app_logger, mock_error_handler)

    assert scraper.config == mock_config
    assert scraper.data_access == mock_data_access
    assert scraper.page_scraper == mock_page_scraper
    assert scraper.app_logger == mock_app_logger
    assert scraper.error_handler == mock_error_handler
    assert isinstance(scraper.team_id_to_abbrev, dict)
    assert isinstance(scraper.team_abbrev_to_id, dict)


def test_load_team_mappings(validation_scraper):
    """Test that team mappings are loaded correctly."""
    assert 'LAL' in validation_scraper.team_abbrev_to_id
    assert 'BOS' in validation_scraper.team_abbrev_to_id
    assert 'SEA' in validation_scraper.team_abbrev_to_id  # Historical team
    assert validation_scraper.team_id_to_abbrev['1610612747'] == 'LAL'


def test_build_date_url(validation_scraper):
    """Test URL building for basketball-reference scoreboard."""
    url = validation_scraper._build_date_url("10/31/2006")

    assert "https://www.basketball-reference.com/boxscores" in url
    assert "month=10" in url
    assert "day=31" in url
    assert "year=2006" in url


def test_extract_team_abbrev_from_url(validation_scraper):
    """Test extracting team abbreviation from URL."""
    url = "https://www.basketball-reference.com/teams/LAL/2024.html"
    abbrev = validation_scraper._extract_team_abbrev_from_url(url)

    assert abbrev == "LAL"


def test_scrape_games_for_date_no_games(validation_scraper):
    """Test scraping when no games are found."""
    validation_scraper.page_scraper.go_to_url = Mock(return_value=True)
    validation_scraper.page_scraper.page_scraper.driver.find_elements = Mock(return_value=[])

    games = validation_scraper._scrape_games_for_date("10/31/2006")

    assert games == []
    validation_scraper.page_scraper.go_to_url.assert_called_once()


def test_scrape_games_for_date_page_load_failure(validation_scraper):
    """Test scraping when page fails to load."""
    validation_scraper.page_scraper.go_to_url = Mock(return_value=False)

    games = validation_scraper._scrape_games_for_date("10/31/2006")

    assert games == []


def test_parse_single_game_from_container_success(validation_scraper):
    """Test parsing a single game from its container."""
    # Create mock container and elements
    mock_container = Mock(spec=WebElement)
    mock_container.text = "Phoenix 106 Final LA Lakers 114"

    mock_visitor_link = Mock(spec=WebElement)
    mock_visitor_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/teams/LAL/2007.html")

    mock_home_link = Mock(spec=WebElement)
    mock_home_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/teams/BOS/2007.html")

    mock_container.find_elements = Mock(return_value=[mock_visitor_link, mock_home_link])

    mock_box_score_link = Mock(spec=WebElement)
    mock_box_score_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/boxscores/200610310BOS.html")

    game_data = validation_scraper._parse_single_game_from_container(
        "10/31/2006",
        mock_container,
        mock_box_score_link
    )

    assert game_data is not None
    assert game_data['DATE'] == "10/31/2006"
    assert game_data['VISITOR_TEAM_ABBREV'] == 'LAL'
    assert game_data['HOME_TEAM_ABBREV'] == 'BOS'
    assert game_data['VISITOR_TEAM_ID'] == '1610612747'
    assert game_data['HOME_TEAM_ID'] == '1610612738'
    assert game_data['VISITOR_SCORE'] == '106'
    assert game_data['HOME_SCORE'] == '114'
    assert game_data['BBREF_GAME_ID'] == '200610310BOS'
    assert game_data['SOURCE'] == 'basketball-reference'


def test_parse_single_game_insufficient_team_links(validation_scraper):
    """Test parsing fails when insufficient team links found."""
    mock_container = Mock(spec=WebElement)
    mock_container.find_elements = Mock(return_value=[Mock()])  # Only 1 link

    mock_box_score_link = Mock(spec=WebElement)

    game_data = validation_scraper._parse_single_game_from_container(
        "10/31/2006",
        mock_container,
        mock_box_score_link
    )

    assert game_data is None


def test_parse_single_game_no_scores_found(validation_scraper):
    """Test parsing fails when scores cannot be extracted."""
    mock_container = Mock(spec=WebElement)
    mock_container.text = "Phoenix Final LA Lakers"  # No scores

    mock_visitor_link = Mock(spec=WebElement)
    mock_visitor_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/teams/LAL/2007.html")

    mock_home_link = Mock(spec=WebElement)
    mock_home_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/teams/BOS/2007.html")

    mock_container.find_elements = Mock(return_value=[mock_visitor_link, mock_home_link])

    mock_box_score_link = Mock(spec=WebElement)
    mock_box_score_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/boxscores/200610310BOS.html")

    game_data = validation_scraper._parse_single_game_from_container(
        "10/31/2006",
        mock_container,
        mock_box_score_link
    )

    assert game_data is None


def test_scrape_validation_data_by_dates_empty_list(validation_scraper):
    """Test scraping with empty date list."""
    df = validation_scraper.scrape_validation_data_by_dates([])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_scrape_and_save_validation_data_no_dates(validation_scraper):
    """Test that scraping returns False when no dates provided."""
    result = validation_scraper.scrape_and_save_validation_data([])

    assert result is None
    validation_scraper.app_logger.structured_log.assert_called()


def test_scrape_and_save_validation_data_empty_dataframe(validation_scraper):
    """Test that empty dataframe is not saved."""
    with patch.object(validation_scraper, 'scrape_validation_data_by_dates') as mock_scrape:
        mock_scrape.return_value = pd.DataFrame()

        validation_scraper.scrape_and_save_validation_data(["10/31/2006"])

        validation_scraper.data_access.save_dataframes.assert_not_called()


def test_scrape_and_save_validation_data_success(validation_scraper):
    """Test successful validation data scraping and saving."""
    mock_df = pd.DataFrame({
        'DATE': ['10/31/2006'],
        'VISITOR_TEAM_ID': ['1610612747'],
        'HOME_TEAM_ID': ['1610612738'],
        'VISITOR_SCORE': ['106'],
        'HOME_SCORE': ['114']
    })

    with patch.object(validation_scraper, 'scrape_validation_data_by_dates') as mock_scrape:
        mock_scrape.return_value = mock_df

        validation_scraper.scrape_and_save_validation_data(["10/31/2006"])

        validation_scraper.data_access.save_dataframes.assert_called_once()
        args = validation_scraper.data_access.save_dataframes.call_args
        assert len(args[0][0]) == 1  # One dataframe
        assert args[0][1][0] == validation_scraper.config.validation_data_file


def test_parse_single_game_unknown_team(validation_scraper):
    """Test parsing with unknown team abbreviation."""
    mock_container = Mock(spec=WebElement)
    mock_container.text = "Unknown Team 106 Final LA Lakers 114"

    mock_visitor_link = Mock(spec=WebElement)
    mock_visitor_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/teams/UNK/2007.html")

    mock_home_link = Mock(spec=WebElement)
    mock_home_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/teams/LAL/2007.html")

    mock_container.find_elements = Mock(return_value=[mock_visitor_link, mock_home_link])

    mock_box_score_link = Mock(spec=WebElement)
    mock_box_score_link.get_attribute = Mock(return_value="https://www.basketball-reference.com/boxscores/200610310LAL.html")

    game_data = validation_scraper._parse_single_game_from_container(
        "10/31/2006",
        mock_container,
        mock_box_score_link
    )

    # Should return None because UNK is not in team mappings
    assert game_data is None
