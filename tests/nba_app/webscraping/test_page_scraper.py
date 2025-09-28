import pytest
from unittest.mock import Mock, patch
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait

from nba_app.webscraping.page_scraper import PageScraper
from platform_core.core.error_handling.error_handler import (
    WebDriverError, PageLoadError, ElementNotFoundError,
    DataExtractionError, DynamicContentLoadError
)

@pytest.fixture
def mock_config():
    config = Mock()
    config.wait_time = 10
    config.page_load_timeout = 30
    config.dynamic_content_timeout = 20
    config.max_retries = 3
    config.retry_delay = 1
    config.no_data_class_name = "no-data"
    return config

@pytest.fixture
def mock_web_driver():
    return Mock()

@pytest.fixture
def scraper(mock_config, mock_web_driver):
    return PageScraper(mock_config, mock_web_driver)

def test_initialization(mock_config, mock_web_driver):
    scraper = PageScraper(mock_config, mock_web_driver)
    assert scraper.config == mock_config
    assert scraper.web_driver == mock_web_driver
    assert isinstance(scraper.wait, WebDriverWait)

def test_go_to_url_success(scraper):
    url = "https://example.com"
    assert scraper.go_to_url(url) == True
    scraper.web_driver.get.assert_called_once_with(url)

def test_go_to_url_timeout(scraper):
    scraper.web_driver.get.side_effect = TimeoutException()
    with pytest.raises(PageLoadError):
        scraper.go_to_url("https://example.com")

def test_go_to_url_webdriver_exception(scraper):
    scraper.web_driver.get.side_effect = WebDriverException()
    with pytest.raises(PageLoadError):
        scraper.go_to_url("https://example.com")

def test_wait_for_dynamic_content_success(scraper):
    locator = (By.ID, "test-id")
    assert scraper.wait_for_dynamic_content(locator) == True

def test_wait_for_dynamic_content_timeout(scraper):
    locator = (By.ID, "test-id")
    scraper.web_driver.page_source = "<html></html>"
    scraper.web_driver.current_url = "https://example.com"
    scraper.web_driver.execute_script.return_value = []
    
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.side_effect = TimeoutException()
        with pytest.raises(DynamicContentLoadError):
            scraper.wait_for_dynamic_content(locator)

def test_handle_pagination(scraper):
    mock_pagination = Mock()
    mock_dropdown = Mock()
    mock_pagination.find_element.return_value = mock_dropdown
    
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.return_value = mock_pagination
        scraper._handle_pagination("pagination-class", "dropdown-class")
        
        mock_dropdown.send_keys.assert_called_once_with("ALL")
        scraper.web_driver.execute_script.assert_called_once()

def test_handle_pagination_no_dropdown(scraper):
    mock_pagination = Mock()
    mock_pagination.find_element.side_effect = NoSuchElementException()
    
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.return_value = mock_pagination
        scraper._handle_pagination("pagination-class", "dropdown-class")
        mock_pagination.find_element.assert_called_once_with(By.CLASS_NAME, "dropdown-class")

def test_get_elements_by_class_success(scraper):
    mock_elements = [Mock(spec=WebElement)]
    
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.return_value = mock_elements
        result = scraper.get_elements_by_class("test-class")
        assert result == mock_elements

def test_get_elements_by_class_with_parent(scraper):
    mock_parent = Mock(spec=WebElement)
    mock_elements = [Mock(spec=WebElement)]
    mock_parent.find_elements.return_value = mock_elements
    
    result = scraper.get_elements_by_class("test-class", mock_parent)
    assert result == mock_elements
    mock_parent.find_elements.assert_called_once_with(By.CLASS_NAME, "test-class")

def test_get_elements_by_class_not_found(scraper):
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.side_effect = TimeoutException()
        with pytest.raises(ElementNotFoundError):
            scraper.get_elements_by_class("test-class")

def test_scrape_page_table_success(scraper):
    mock_table = Mock(spec=WebElement)
    
    with patch.object(scraper, 'go_to_url') as mock_go_to_url:
        with patch.object(scraper, 'get_elements_by_class') as mock_get_elements:
            mock_go_to_url.return_value = True
            mock_get_elements.return_value = [mock_table]
            
            result = scraper.scrape_page_table(
                "https://example.com",
                "table-class",
                "pagination-class",
                "dropdown-class"
            )
            
            assert result == mock_table

def test_scrape_page_table_no_data(scraper):
    with patch.object(scraper, 'go_to_url') as mock_go_to_url:
        with patch.object(scraper, 'get_elements_by_class') as mock_get_elements:
            mock_go_to_url.return_value = True
            
            mock_get_elements.side_effect = [
                ElementNotFoundError("Table not found"),  # table_class check
                [Mock()]  # no_data_class_name check
            ]
            
            result = scraper.scrape_page_table(
                "https://example.com",
                "table-class",
                "pagination-class",
                "dropdown-class"
            )
            
            assert result is None
            mock_get_elements.assert_any_call(scraper.config.no_data_class_name)

def test_safe_wait_and_click_success(scraper):
    mock_element = Mock()
    locator = (By.ID, "test-id")
    
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.return_value = mock_element
        scraper._safe_wait_and_click(locator)
        mock_element.click.assert_called_once()

def test_safe_wait_and_click_failure(scraper):
    locator = (By.ID, "test-id")
    
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.side_effect = TimeoutException()
        with pytest.raises(ElementNotFoundError):
            scraper._safe_wait_and_click(locator)

def test_get_links_by_class_success(scraper):
    mock_links = [Mock(spec=WebElement)]
    
    with patch('selenium.webdriver.support.wait.WebDriverWait.until') as mock_until:
        mock_until.return_value = mock_links
        result = scraper.get_links_by_class("test-class")
        assert result == mock_links

def test_get_links_by_class_with_parent(scraper):
    mock_parent = Mock(spec=WebElement)
    mock_links = [Mock(spec=WebElement)]
    mock_parent.find_elements.return_value = mock_links
    
    result = scraper.get_links_by_class("test-class", mock_parent)
    assert result == mock_links