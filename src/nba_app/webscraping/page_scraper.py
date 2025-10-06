"""
page_scraper.py

This module provides a PageScraper class for web scraping operations using Selenium.
It implements the BasePageScraper interface and includes enhanced logging and error handling.
"""

import logging
import time
from typing import Optional, List, Tuple

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException, WebDriverException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver

from .base_scraper_classes import BasePageScraper
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory
from ml_framework.core.app_logging import log_performance, AppLogger

class PageScraper(BasePageScraper):
    @log_performance
    def __init__(self, config: BaseConfigManager, web_driver: WebDriver, app_logger: AppLogger, error_handler: ErrorHandlerFactory) -> None:
        """
        Initialize the PageScraper with configuration and web driver.

        Args:
            config (BaseConfigManager): Configuration object.
            web_driver (WebDriver): Web driver object.
            app_logger (AppLogger): Application logger instance.
            error_handler (ErrorHandlerFactory): Error handler factory instance.

        Raises:
            WebDriverError: If there's an issue initializing the WebDriver.
        """
        self.config = config
        self.web_driver = web_driver
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.wait = WebDriverWait(self.web_driver, self.config.wait_time)
        self.app_logger.structured_log(logging.INFO, "PageScraper initialized successfully",
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
            self.app_logger.structured_log( logging.ERROR, "WebDriver is not initialized")
            return False

        self.app_logger.structured_log( logging.INFO, "Navigating to URL", url=url)

        try:
            self.web_driver.get(url)


            self.app_logger.structured_log( logging.INFO, "Navigation to URL completed successfully", url=url)
            return True


        except TimeoutException as e:
            raise self.error_handler.create_error_handler('page_load', "Timeout occurred while loading URL", url=url, timeout=self.config.page_load_timeout)

        except WebDriverException as e:
            raise self.error_handler.create_error_handler('page_load', "WebDriver exception occurred", url=url, error_message=str(e))

        except Exception as e:
            raise self.error_handler.create_error_handler('page_load', f"Unexpected error navigating to URL", url=url, error_message=str(e))

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

        self.app_logger.structured_log( logging.INFO, "Waiting for dynamic content to load", locator=locator, timeout=timeout)

        try:
            WebDriverWait(self.web_driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            self.app_logger.structured_log( logging.INFO, "Dynamic content loaded successfully", locator=locator)
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


            raise self.error_handler.create_error_handler('dynamic_content_load', "Timeout while waiting for dynamic content", **error_info)
        except Exception as e:
            raise self.error_handler.create_error_handler('dynamic_content_load', "Error while waiting for dynamic content",
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
        self.app_logger.structured_log( logging.INFO, "Handling pagination",
                       pagination_class=pagination_class, dropdown_class=dropdown_class)
        try:
            pagination = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, pagination_class)))
            self.app_logger.structured_log( logging.INFO, "Pagination element found")
            try:
                page_dropdown = pagination.find_element(By.CLASS_NAME, dropdown_class)
                self.app_logger.structured_log( logging.INFO, "Dropdown element found")
                page_dropdown.send_keys("ALL")
                time.sleep(3)
                self.web_driver.execute_script('arguments[0].click()', page_dropdown)
                self.app_logger.structured_log( logging.INFO, "Selected 'ALL' in the pagination dropdown")

            except NoSuchElementException:
                raise self.error_handler.create_error_handler('element_not_found', "No dropdown found in pagination element",
                                           pagination_class=pagination_class,
                                           dropdown_class=dropdown_class)
        except:
            self.app_logger.structured_log( logging.INFO, "No pagination found on the page",
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
        self.app_logger.structured_log( logging.INFO, "Retrieving elements by class",
                       class_name=class_name, has_parent=parent_element is not None)
        for attempt in range(self.config.max_retries):
            try:
                if parent_element:
                    elements = parent_element.find_elements(By.CLASS_NAME, class_name)
                else:
                    elements = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, class_name)))
                if not elements:
                    self.app_logger.structured_log( logging.INFO, "No elements found",
                                   class_name=class_name, attempt=attempt+1)
                    return None
                self.app_logger.structured_log( logging.INFO, "Elements found",
                               class_name=class_name, element_count=len(elements))
                return elements
            except (TimeoutException, NoSuchElementException, StaleElementReferenceException):
                self.app_logger.structured_log( logging.INFO, "Element not found or stale",
                               class_name=class_name, attempt=attempt+1)
            except Exception as e:
                self.app_logger.structured_log( logging.INFO, "Unexpected error occurred",
                               class_name=class_name, attempt=attempt+1,
                               error_message=str(e), error_type=type(e).__name__)
            if attempt < self.config.max_retries - 1:
                # Incremental backoff: delay increases with each retry attempt
                delay = self.config.retry_delay * (attempt + 1)
                self.app_logger.structured_log( logging.INFO, "Waiting before retry",
                               class_name=class_name, attempt=attempt+1, delay_seconds=delay)
                time.sleep(delay)
        raise self.error_handler.create_error_handler('element_not_found', f"Failed to retrieve elements with class {class_name} after {self.config.max_retries} attempts")

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
        self.app_logger.structured_log( logging.INFO, "Starting page scrape", url=url)
        try:
            if not self.go_to_url(url):
                raise self.error_handler.create_error_handler('page_load', "Could not navigate to URL", url=url)

            # see if a table is present, if not, check for a no data message (which liekly means there were no games played for these parameters)
            try:
                if self.get_elements_by_class(table_class) is None:
                    self.app_logger.structured_log( logging.INFO, "No table found, checking for no data message", url=url)
            except:
                try:
                    if self.get_elements_by_class(self.config.no_data_class_name) is not None:
                        self.app_logger.structured_log( logging.INFO, "No data message found", url=url)
                        return None
                except:
                    raise self.error_handler.create_error_handler('dynamic_content_load', "Table did not load within the specified timeout",
                                                  table_class=table_class)


            self._handle_pagination(pagination_class, dropdown_class)

            tables = self.get_elements_by_class(table_class)
            if tables is None:
                raise self.error_handler.create_error_handler('element_not_found', "Table not found after pagination",
                                           table_class=table_class)

            data_table = tables[0]
            if data_table is not None:
                self.app_logger.structured_log( logging.INFO, "Table found and scraped successfully")
                return data_table
            else:
                raise self.error_handler.create_error_handler('element_not_found', "Table not found after retrieval",
                                           table_class=table_class)
        except Exception as e:
            # Check if it's already one of our error types (has app_logger)
            if hasattr(e, 'app_logger'):
                raise
            raise self.error_handler.create_error_handler('data_extraction', "Unexpected error during page scrape",
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
        self.app_logger.structured_log( logging.INFO, "Attempting to click element", locator=locator)
        for attempt in range(self.config.max_retries):
            try:
                element = self.wait.until(EC.element_to_be_clickable(locator))
                element.click()
                self.app_logger.structured_log( logging.DEBUG, "Successfully clicked element",
                               locator=locator, attempt=attempt+1)
                return
            except (StaleElementReferenceException, TimeoutException):
                self.app_logger.structured_log( logging.DEBUG, "Failed to click element",
                               locator=locator, attempt=attempt+1)
                if attempt == self.config.max_retries - 1:
                    self.app_logger.structured_log( logging.ERROR, "Element not clickable after all attempts",
                                   locator=locator, max_retries=self.config.max_retries)
                    raise self.error_handler.create_error_handler('element_not_found', f"Element {locator} was not clickable after {self.config.max_retries} attempts")
                self.app_logger.structured_log( logging.DEBUG, "Retrying click attempt",
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
        self.app_logger.structured_log( logging.INFO, "Retrieving link elements by class",
                       class_name=class_name, has_parent=parent_element is not None)

        for attempt in range(self.config.max_retries):
            try:
                xpath = f".//a[@class='{class_name}']" if parent_element else f"//a[@class='{class_name}']"
                if parent_element:
                    elements = parent_element.find_elements(By.XPATH, xpath)
                else:
                    elements = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, xpath)))

                if not elements:
                    self.app_logger.structured_log( logging.INFO, "No link elements found",
                                   class_name=class_name, attempt=attempt+1)
                    return None

                self.app_logger.structured_log( logging.INFO, "Link elements found",
                               class_name=class_name, element_count=len(elements))
                return elements

            except (TimeoutException, NoSuchElementException, StaleElementReferenceException):
                self.app_logger.structured_log( logging.INFO, "Link elements not found or stale",
                               class_name=class_name, attempt=attempt+1)

            except Exception as e:
                self.app_logger.structured_log( logging.INFO, "Unexpected error occurred",
                               class_name=class_name, attempt=attempt+1,
                               error_message=str(e), error_type=type(e).__name__)

            if attempt < self.config.max_retries - 1:
                # Incremental backoff: delay increases with each retry attempt
                delay = self.config.retry_delay * (attempt + 1)
                self.app_logger.structured_log( logging.INFO, "Waiting before retry",
                               class_name=class_name, attempt=attempt+1, delay_seconds=delay)
                time.sleep(delay)

        raise self.error_handler.create_error_handler('element_not_found', f"Failed to retrieve link elements with class {class_name} after {self.config.max_retries} attempts")


                
