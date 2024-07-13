import pandas as pd
from unittest.mock import Mock, patch
import logging
import time
from typing import List, Tuple, Optional

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

# Import the PageScraper class
from .page_scraper import PageScraper 

@pytest.fixture
def mock_driver() -> Mock:
    """
    Creates a mock WebDriver object.

    Returns:
        Mock: A mock object simulating a WebDriver.
    """
    return Mock()

@pytest.fixture
def scraper(mock_driver: Mock) -> PageScraper:
    """
    Creates a PageScraper instance with a mock driver.

    Args:
        mock_driver (Mock): A mock WebDriver object.

    Returns:
        PageScraper: An instance of PageScraper with the mock driver.
    """
    return PageScraper(mock_driver, log_level=logging.ERROR)

def test_init(scraper: PageScraper) -> None:
    """
    Tests the initialization of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
    """
    assert isinstance(scraper.wait, WebDriverWait)
    assert scraper.logger.level == logging.ERROR

def test_go_to_url(scraper: PageScraper) -> None:
    """
    Tests the go_to_url method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
    """
    url = "https://example.com"
    scraper.go_to_url(url)
    scraper.driver.get.assert_called_once_with(url)

@pytest.mark.parametrize("dropdown_exists", [True, False])
def test_handle_pagination(scraper: PageScraper, dropdown_exists: bool) -> None:
    """
    Tests the handle_pagination method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
        dropdown_exists (bool): Whether the dropdown element exists.
    """
    mock_pagination = Mock()
    scraper.wait.until = Mock(return_value=mock_pagination)

    if dropdown_exists:
        mock_dropdown = Mock()
        mock_pagination.find_element.return_value = mock_dropdown
        with patch('page_scraper.Select') as mock_select:
            mock_select.return_value = mock_dropdown
            scraper.handle_pagination("pagination-class", "dropdown-class")
            mock_dropdown.select_by_visible_text.assert_called_once_with("ALL")
    else:
        mock_pagination.find_element.side_effect = NoSuchElementException()
        scraper.handle_pagination("pagination-class", "dropdown-class")
    
    scraper.wait.until.assert_called()

@pytest.mark.parametrize("table_found", [True, False])
def test_get_table(scraper: PageScraper, table_found: bool) -> None:
    """
    Tests the get_table method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
        table_found (bool): Whether the table element is found.
    """
    if table_found:
        mock_table = Mock()
        scraper.wait.until = Mock(return_value=mock_table)
        result = scraper.get_table("table-class")
        assert result == mock_table
    else:
        scraper.wait.until = Mock(side_effect=TimeoutException())
        result = scraper.get_table("table-class")
        assert result is None
    
    scraper.wait.until.assert_called_once()

def test_convert_table_df(scraper: PageScraper) -> None:
    """
    Tests the convert_table_df method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
    """
    mock_table = Mock()
    mock_table.get_attribute.return_value = "<table></table>"
    mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    with patch('page_scraper.pd.read_html', return_value=[mock_df]):
        result = scraper.convert_table_df(mock_table)
        pd.testing.assert_frame_equal(result, mock_df)

def test_get_table_links(scraper: PageScraper) -> None:
    """
    Tests the get_table_links method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
    """
    mock_table = Mock()
    mock_links = [Mock(), Mock()]
    mock_table.find_elements.return_value = mock_links

    result = scraper.get_table_links(mock_table, "link-class")

    assert result == mock_links
    mock_table.find_elements.assert_called_once_with(By.CLASS_NAME, "link-class")

def test_extract_hrefs(scraper: PageScraper) -> None:
    """
    Tests the extract_hrefs method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
    """
    mock_links = [Mock(), Mock()]
    mock_links[0].get_attribute.return_value = "http://example1.com"
    mock_links[1].get_attribute.return_value = "http://example2.com"

    result = scraper.extract_hrefs(mock_links)

    assert result == ["http://example1.com", "http://example2.com"]

def test_get_elements_by_class(scraper: PageScraper) -> None:
    """
    Tests the get_elements_by_class method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
    """
    mock_elements = [Mock(), Mock()]
    scraper.wait.until = Mock(return_value=mock_elements)

    result = scraper.get_elements_by_class("some-class")

    assert result == mock_elements
    scraper.wait.until.assert_called_once()

def test_scrape_page_table(scraper: PageScraper) -> None:
    """
    Tests the scrape_page_table method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
    """
    mock_table = Mock()
    mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    mock_links = [Mock(), Mock()]
    mock_hrefs = ["http://example1.com", "http://example2.com"]

    with patch.multiple(PageScraper,
                        go_to_url=Mock(),
                        handle_pagination=Mock(),
                        get_table=Mock(return_value=mock_table),
                        convert_table_df=Mock(return_value=mock_df),
                        get_table_links=Mock(return_value=mock_links),
                        extract_hrefs=Mock(return_value=mock_hrefs)):

        result = scraper.scrape_page_table("http://example.com", "table-class", 
                                           "pagination-class", "dropdown-class", "link-class")

        assert result == (mock_df, mock_links, mock_hrefs)
        scraper.go_to_url.assert_called_once_with("http://example.com")
        scraper.handle_pagination.assert_called_once_with("pagination-class", "dropdown-class")
        scraper.get_table.assert_called_once_with("table-class")
        scraper.convert_table_df.assert_called_once_with(mock_table)
        scraper.get_table_links.assert_called_once_with(mock_table, "link-class")
        scraper.extract_hrefs.assert_called_once_with(mock_links)

@pytest.mark.parametrize("success_on_attempt", [1, 2, 3, None])
def test_safe_wait_and_click(scraper: PageScraper, success_on_attempt: Optional[int]) -> None:
    """
    Tests the safe_wait_and_click method of PageScraper.

    Args:
        scraper (PageScraper): An instance of PageScraper.
        success_on_attempt (Optional[int]): The attempt number on which the click succeeds, or None if it always fails.
    """
    mock_element = Mock()

    if success_on_attempt is not None:
        side_effects = [TimeoutException()] * (success_on_attempt - 1) + [mock_element]
        scraper.wait.until = Mock(side_effect=side_effects)
        with patch('time.sleep', Mock()):
            scraper.safe_wait_and_click((By.ID, "some-id"))
        assert scraper.wait.until.call_count == success_on_attempt
        mock_element.click.assert_called_once()
    else:
        scraper.wait.until = Mock(side_effect=TimeoutException())
        with pytest.raises(TimeoutException):
            scraper.safe_wait_and_click((By.ID, "some-id"), max_attempts=3)
        assert scraper.wait.until.call_count == 3
