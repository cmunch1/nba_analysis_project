"""
page_scraper.py

This module provides a PageScraper class for web scraping operations using Selenium.
It implements the AbstractPageScraper interface and includes enhanced logging and error handling.
"""

import sys
import logging
import time
from typing import Optional, List, Tuple

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException, WebDriverException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver

from .abstract_scraper_classes import AbstractPageScraper
from ..config.config import AbstractConfig
from ..error_handling.custom_exceptions import WebDriverError, PageLoadError, ElementNotFoundError, DataExtractionError, DynamicContentLoadError
from ..logging.logging_utils import log_performance, structured_log

logger = logging.getLogger(__name__)

class PageScraper(AbstractPageScraper):
    @log_performance
    def __init__(self, config: AbstractConfig, web_driver: WebDriver) -> None:
        """
        Initialize the PageScraper with configuration and web driver.

        Args:
            config (AbstractConfig): Configuration object.
            web_driver (WebDriver): Web driver object.

        Raises:
            WebDriverError: If there's an issue initializing the WebDriver.
        """
        self.config = config
        self.web_driver = web_driver
        self.wait = WebDriverWait(self.web_driver, self.config.wait_time)
        structured_log(logger, logging.INFO, "PageScraper initialized successfully",
                       config_type=type(config).__name__,
                       web_driver_type=type(web_driver).__name__)

    @log_performance
    def go_to_url(self, url: str) -> bool:
        """
        Navigate to the specified URL with enhanced error handling.

        Args:
            url (str): The URL to navigate to.

        Returns:
            bool: True if navigation was successful, False otherwise.

        Raises:
            PageLoadError: If there's an error loading the page.
        """
        if self.web_driver is None:
            structured_log(logger, logging.ERROR, "WebDriver is not initialized")
            return False

        structured_log(logger, logging.INFO, "Navigating to URL", url=url)
        
        try:
            self.web_driver.get(url)
            

            structured_log(logger, logging.INFO, "Navigation to URL completed successfully", url=url)
            return True
        
        except TimeoutException as e:
            raise PageLoadError("Timeout occurred while loading URL", url=url, timeout=self.config.page_load_timeout)
        
        except WebDriverException as e:
            raise PageLoadError("WebDriver exception occurred", url=url, error_message=str(e))
        
        except Exception as e:
            raise PageLoadError(f"Unexpected error navigating to URL", url=url, error_message=str(e))

    @log_performance
    def wait_for_dynamic_content(self, locator: Tuple[str, str], timeout: int = None) -> bool:
        """
        Wait for dynamic content to load on the page.

        Args:
            locator (Tuple[str, str]): A tuple containing the locator strategy and locator value.
            timeout (int, optional): The maximum time to wait for the content to load.

        Returns:
            bool: True if the content loaded successfully, False otherwise.

        Raises:
            DynamicContentLoadError: If the content fails to load within the specified timeout.
        """
        if timeout is None:
            timeout = self.config.dynamic_content_timeout

        structured_log(logger, logging.INFO, "Waiting for dynamic content to load", locator=locator, timeout=timeout)

        try:
            WebDriverWait(self.web_driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            structured_log(logger, logging.INFO, "Dynamic content loaded successfully", locator=locator)
            return True
        except TimeoutException:
            # Collect additional information about the page state
            page_source = self.web_driver.page_source
            current_url = self.web_driver.current_url
            js_errors = self.web_driver.execute_script("var err = window.JSErrors || []; return err;")
            
            error_info = {
                "locator": locator,
                "timeout": timeout,
                "current_url": current_url,
                "js_errors": js_errors,
                "page_source_excerpt": page_source[:1000]  # First 1000 characters of page source
            }
            
            raise DynamicContentLoadError("Timeout while waiting for dynamic content", **error_info)
        except Exception as e:
            raise DynamicContentLoadError("Error while waiting for dynamic content", 
                                          locator=locator, error_message=str(e))

    @log_performance
    def _handle_pagination(self, pagination_class: str, dropdown_class: str) -> None:
        """
        Handle pagination on the page by selecting 'ALL' in the dropdown if available.

        Args:
            pagination_class (str): The class name of the pagination element.
            dropdown_class (str): The class name of the dropdown element within the pagination.

        Raises:
            ElementNotFoundError: If the required elements are not found on the page.
        """
        structured_log(logger, logging.INFO, "Handling pagination",
                       pagination_class=pagination_class, dropdown_class=dropdown_class)
        try:
            pagination = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, pagination_class)))
            structured_log(logger, logging.INFO, "Pagination element found")
            try:
                page_dropdown = pagination.find_element(By.CLASS_NAME, dropdown_class)
                structured_log(logger, logging.INFO, "Dropdown element found")
                page_dropdown.send_keys("ALL")
                time.sleep(3)
                self.web_driver.execute_script('arguments[0].click()', page_dropdown)
                structured_log(logger, logging.INFO, "Selected 'ALL' in the pagination dropdown")

            except NoSuchElementException:
                raise ElementNotFoundError("No dropdown found in pagination element", 
                                           pagination_class=pagination_class, 
                                           dropdown_class=dropdown_class)
        except:
            structured_log(logger, logging.INFO, "No pagination found on the page",
                           pagination_class=pagination_class, dropdown_class=dropdown_class)

    @log_performance
    def get_elements_by_class(self, class_name: str, parent_element: Optional[WebElement] = None) -> Optional[List[WebElement]]:
        """
        Retrieve elements by class name, optionally within a parent element.

        Args:
            class_name (str): The class name to search for.
            parent_element (Optional[WebElement]): The parent element to search within, if any.

        Returns:
            Optional[List[WebElement]]: A list of found elements, or None if no elements were found.

        Raises:
            ElementNotFoundError: If the elements are not found after multiple attempts.
        """
        structured_log(logger, logging.INFO, "Retrieving elements by class",
                       class_name=class_name, has_parent=parent_element is not None)
        for attempt in range(self.config.max_retries):
            try:
                if parent_element:
                    elements = parent_element.find_elements(By.CLASS_NAME, class_name)
                else:
                    elements = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, class_name)))
                if not elements:
                    structured_log(logger, logging.INFO, "No elements found",
                                   class_name=class_name, attempt=attempt+1)
                    return None
                structured_log(logger, logging.INFO, "Elements found",
                               class_name=class_name, element_count=len(elements))
                return elements
            except (TimeoutException, NoSuchElementException, StaleElementReferenceException):
                structured_log(logger, logging.INFO, "Element not found or stale",
                               class_name=class_name, attempt=attempt+1)
            except Exception as e:
                structured_log(logger, logging.INFO, "Unexpected error occurred",
                               class_name=class_name, attempt=attempt+1,
                               error_message=str(e), error_type=type(e).__name__)
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay)
        raise ElementNotFoundError(f"Failed to retrieve elements with class {class_name} after {self.config.max_retries} attempts")

    @log_performance
    def scrape_page_table(self, url: str, table_class: str, pagination_class: str, dropdown_class: str) -> Optional[WebElement]:
        """
        Scrape a table from a web page, handling pagination if necessary.

        Args:
            url (str): The URL of the page to scrape.
            table_class (str): The class name of the table to scrape.
            pagination_class (str): The class name of the pagination element.
            dropdown_class (str): The class name of the dropdown element within the pagination.

        Returns:
            Optional[WebElement]: The scraped table as a WebElement, or None if scraping failed.

        Raises:
            PageLoadError: If there's an error loading the page.
            DynamicContentLoadError: If the dynamic content (table) fails to load.
            ElementNotFoundError: If the required elements are not found on the page.
            DataExtractionError: If there's an unexpected error during scraping.
        """
        structured_log(logger, logging.INFO, "Starting page scrape", url=url)
        try:
            if not self.go_to_url(url):
                raise PageLoadError("Could not navigate to URL", url=url)
            
            # see if a table is present, if not, check for a no data message (which liekly means there were no games played for these parameters)
            try:
                if self.get_elements_by_class(table_class) is None:
                    structured_log(logger, logging.INFO, "No table found, checking for no data message", url=url)
            except:
                try:
                    if self.get_elements_by_class(self.config.no_data_class_name) is not None:
                        structured_log(logger, logging.INFO, "No data message found", url=url)
                        return None
                except:
                    raise DynamicContentLoadError("Table did not load within the specified timeout", 
                                                  table_class=table_class)


            self._handle_pagination(pagination_class, dropdown_class)
            
            tables = self.get_elements_by_class(table_class)
            if tables is None:
                raise ElementNotFoundError("Table not found after pagination", 
                                           table_class=table_class)
            
            data_table = tables[0] 
            if data_table is not None:
                structured_log(logger, logging.INFO, "Table found and scraped successfully")
                return data_table
            else:
                raise ElementNotFoundError("Table not found after retrieval", 
                                           table_class=table_class)
        except (PageLoadError, DynamicContentLoadError, ElementNotFoundError):
            raise
        except Exception as e:
            raise DataExtractionError("Unexpected error during page scrape", 
                                      url=url, error_message=str(e))

    @log_performance
    def _safe_wait_and_click(self, locator: Tuple[str, str]) -> None:
        """
        Safely wait for an element to be clickable and then click it, with retries.

        Args:
            locator (Tuple[str, str]): A tuple containing the locator strategy and locator value.

        Raises:
            ElementNotFoundError: If the element is not clickable after all retry attempts.
        """
        structured_log(logger, logging.INFO, "Attempting to click element", locator=locator)
        for attempt in range(self.config.max_retries):
            try:
                element = self.wait.until(EC.element_to_be_clickable(locator))
                element.click()
                structured_log(logger, logging.DEBUG, "Successfully clicked element",
                               locator=locator, attempt=attempt+1)
                return
            except (StaleElementReferenceException, TimeoutException):
                structured_log(logger, logging.DEBUG, "Failed to click element",
                               locator=locator, attempt=attempt+1)
                if attempt == self.config.max_retries - 1:
                    structured_log(logger, logging.ERROR, "Element not clickable after all attempts",
                                   locator=locator, max_retries=self.config.max_retries)
                    raise ElementNotFoundError(f"Element {locator} was not clickable after {self.config.max_retries} attempts")
                structured_log(logger, logging.DEBUG, "Retrying click attempt",
                               locator=locator, retry_delay=self.config.retry_delay)
                time.sleep(self.config.retry_delay)

    @log_performance
    def get_links_by_class(self, class_name: str, parent_element: Optional[WebElement] = None) -> Optional[List[WebElement]]:
        """
        Retrieve link elements (a tags) by class name, optionally within a parent element.

        Args:
            class_name (str): The class name to search for.
            parent_element (Optional[WebElement]): The parent element to search within, if any.

        Returns:
            Optional[List[WebElement]]: A list of found link elements, or None if no elements were found.

        Raises:
            ElementNotFoundError: If the elements are not found after multiple attempts.
        """
        structured_log(logger, logging.INFO, "Retrieving link elements by class",
                       class_name=class_name, has_parent=parent_element is not None)
        
        for attempt in range(self.config.max_retries):
            try:
                xpath = f".//a[@class='{class_name}']" if parent_element else f"//a[@class='{class_name}']"
                if parent_element:
                    elements = parent_element.find_elements(By.XPATH, xpath)
                else:
                    elements = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, xpath)))
                
                if not elements:
                    structured_log(logger, logging.INFO, "No link elements found",
                                   class_name=class_name, attempt=attempt+1)
                    return None
                
                structured_log(logger, logging.INFO, "Link elements found",
                               class_name=class_name, element_count=len(elements))
                return elements
                
            except (TimeoutException, NoSuchElementException, StaleElementReferenceException):
                structured_log(logger, logging.INFO, "Link elements not found or stale",
                               class_name=class_name, attempt=attempt+1)
                
            except Exception as e:
                structured_log(logger, logging.INFO, "Unexpected error occurred",
                               class_name=class_name, attempt=attempt+1,
                               error_message=str(e), error_type=type(e).__name__)
                
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay)
                
        raise ElementNotFoundError(f"Failed to retrieve link elements with class {class_name} after {self.config.max_retries} attempts")


                
