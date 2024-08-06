"""
test_page_scraper.py

This module contains unit tests for the PageScraper class defined in page_scraper.py.
It uses pytest for test organization and execution, and unittest.mock for mocking
external dependencies.

The tests cover all methods of the PageScraper class, including:
- Initialization
- URL navigation
- Pagination handling
- Element retrieval
- Table scraping
- Safe clicking with retries

These tests ensure the reliability and correctness of the PageScraper class,
which is crucial for the web scraping operations in the NBA analysis project.
"""

import pytest
from unittest.mock import Mock, patch
import logging
import time
from typing import List, Tuple, Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from src.webscraping.page_scraper import PageScraper
from src.config.config import config

@pytest.fixture
def mock_driver() -> Mock:
    return Mock()

@pytest.fixture
def scraper(mock_driver: Mock) -> PageScraper:
    return PageScraper(mock_driver)

def test_init(scraper: PageScraper) -> None:
    assert isinstance(scraper.driver, WebDriver)
    assert isinstance(scraper.wait, WebDriverWait)
    assert isinstance(scraper.logger, logging.Logger)

def test_go_to_url(scraper: PageScraper) -> None:
    url = "https://example.com"
    scraper.driver.execute_script.return_value = 'complete'
    
    result = scraper.go_to_url(url)
    
    assert result is True
    scraper.driver.get.assert_called_once_with(url)
    scraper.driver.execute_script.assert_called_once_with('return document.readyState')

def test_go_to_url_failure(scraper: PageScraper) -> None:
    url = "https://example.com"
    scraper.driver.get.side_effect = TimeoutException()
    
    result = scraper.go_to_url(url)
    
    assert result is False

def test_handle_pagination(scraper: PageScraper) -> None:
    mock_pagination = Mock()
    mock_dropdown = Mock()
    scraper.wait.until.return_value = mock_pagination
    mock_pagination.find_element.return_value = mock_dropdown
    
    scraper.handle_pagination("pagination-class", "dropdown-class")
    
    mock_pagination.find_element.assert_called_once_with(By.CLASS_NAME, "dropdown-class")
    mock_dropdown.send_keys.assert_called_once_with("ALL")
    scraper.driver.execute_script.assert_called_once_with('arguments[0].click()', mock_dropdown)

def test_handle_pagination_no_dropdown(scraper: PageScraper) -> None:
    mock_pagination = Mock()
    scraper.wait.until.return_value = mock_pagination
    mock_pagination.find_element.side_effect = NoSuchElementException()
    
    scraper.handle_pagination("pagination-class", "dropdown-class")
    
    mock_pagination.find_element.assert_called_once_with(By.CLASS_NAME, "dropdown-class")

def test_get_elements_by_class(scraper: PageScraper) -> None:
    mock_elements = [Mock(), Mock()]
    scraper.wait.until.return_value = mock_elements
    
    result = scraper.get_elements_by_class("some-class")
    
    assert result == mock_elements
    scraper.wait.until.assert_called_once()

def test_get_elements_by_class_with_parent(scraper: PageScraper) -> None:
    mock_parent = Mock()
    mock_elements = [Mock(), Mock()]
    mock_parent.find_elements.return_value = mock_elements
    
    result = scraper.get_elements_by_class("some-class", mock_parent)
    
    assert result == mock_elements
    mock_parent.find_elements.assert_called_once_with(By.CLASS_NAME, "some-class")

def test_get_elements_by_class_not_found(scraper: PageScraper) -> None:
    scraper.wait.until.side_effect = TimeoutException()
    
    result = scraper.get_elements_by_class("some-class")
    
    assert result is None

def test_scrape_page_table(scraper: PageScraper) -> None:
    url = "https://example.com"
    mock_table = Mock()
    
    with patch.object(scraper, 'go_to_url', return_value=True), \
         patch.object(scraper, 'handle_pagination'), \
         patch.object(scraper, 'get_elements_by_class', return_value=[mock_table]):
        
        result = scraper.scrape_page_table(url, "table-class", "pagination-class", "dropdown-class")
        
        assert result == mock_table
        scraper.go_to_url.assert_called_once_with(url)
        scraper.handle_pagination.assert_called_once_with("pagination-class", "dropdown-class")
        scraper.get_elements_by_class.assert_called_once_with("table-class")

def test_scrape_page_table_failure(scraper: PageScraper) -> None:
    url = "https://example.com"
    
    with patch.object(scraper, 'go_to_url', return_value=False):
        result = scraper.scrape_page_table(url, "table-class", "pagination-class", "dropdown-class")
        
        assert result is None
        scraper.go_to_url.assert_called_once_with(url)

def test_safe_wait_and_click(scraper: PageScraper) -> None:
    mock_element = Mock()
    scraper.wait.until.return_value = mock_element
    
    scraper.safe_wait_and_click((By.ID, "some-id"))
    
    scraper.wait.until.assert_called_once()
    mock_element.click.assert_called_once()

def test_safe_wait_and_click_retry(scraper: PageScraper) -> None:
    mock_element = Mock()
    scraper.wait.until.side_effect = [StaleElementReferenceException(), mock_element]
    
    with patch('time.sleep'):
        scraper.safe_wait_and_click((By.ID, "some-id"))
    
    assert scraper.wait.until.call_count == 2
    mock_element.click.assert_called_once()

def test_safe_wait_and_click_failure(scraper: PageScraper) -> None:
    scraper.wait.until.side_effect = TimeoutException()
    
    with pytest.raises(TimeoutException), patch('time.sleep'):
        scraper.safe_wait_and_click((By.ID, "some-id"))
    
    assert scraper.wait.until.call_count == config.max_retries


