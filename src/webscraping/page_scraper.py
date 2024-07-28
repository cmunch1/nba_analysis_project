import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver

import pandas as pd
from typing import Optional, List, Tuple
import time

class PageScraper:
    """
    A class for scraping web pages using Selenium WebDriver.

    This class provides methods for navigating to URLs, handling pagination,
    extracting table data, and retrieving elements and sub-elements.

    Attributes:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        wait (WebDriverWait): A WebDriverWait instance for waiting for elements.
        logger (logging.Logger): Logger instance for this class.
    """

    def __init__(self, driver: webdriver.Chrome, wait_time: int = 10, log_level: int = logging.INFO):
        """
        Initialize the PageScraper with a WebDriver instance.

        Args:
            driver (webdriver.Chrome): The Selenium WebDriver instance to use for scraping.
            wait_time (int): The maximum time to wait for elements to be present. Defaults to 10 seconds.
            log_level (int): The logging level to use. Defaults to logging.INFO.
        """
        self.driver = driver
        self.wait = WebDriverWait(self.driver, wait_time)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info("PageScraper initialized")



    def go_to_url(self, url: str) -> bool:
        """
        Navigate to the given URL and check if the page loaded correctly.

        Args:
            url (str): The URL to navigate to.

        Returns:
            bool: True if navigation was successful and page loaded, False otherwise.
        """
        self.logger.info(f"Navigating to URL: {url}")
        try:
            self.driver.get(url)
            # Check if the page has finished loading
            self.wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
            self.logger.debug(f"Navigation to {url} completed successfully")
            return True
        except TimeoutException:
            self.logger.warning(f"Timeout occurred while loading {url}")
            return False
        except Exception as e:
            self.logger.error(f"Error navigating to {url}: {e}")
            return False
            

    def handle_pagination(self, pagination_class: str, dropdown_class: str) -> None:
        self.logger.info(f"Handling pagination. Pagination class: {pagination_class}, Dropdown class: {dropdown_class}")
        try:
            pagination = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, pagination_class)))
            self.logger.debug("Pagination element found")
            try:
                page_dropdown = pagination.find_element(By.CLASS_NAME, dropdown_class)
                self.logger.debug("Dropdown element found")
                page_dropdown.send_keys("ALL")
                time.sleep(3)
                self.driver.execute_script('arguments[0].click()', page_dropdown)
                self.logger.info("Selected 'ALL' in the pagination dropdown")
                
                # Wait for the page to update after selecting "ALL"
                self.wait.until(EC.staleness_of(pagination))
                self.logger.debug("Original pagination element became stale")
                
                # Wait for the new content to load
                self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, pagination_class)))
                self.logger.debug("New pagination element loaded")
                
            except NoSuchElementException:
                self.logger.warning("No dropdown found in pagination element")
        except TimeoutException:
            self.logger.warning("No pagination found on the page")

            

    def get_elements_by_class(self, class_name: str, parent_element: Optional[WebElement] = None, max_retries: int = 3, retry_delay: int = 2) -> Optional[List[WebElement]]:
        """
        Retrieve all elements or sub-elements with the given class name, with error trapping and retries.

        Args:
            class_name (str): The class name to search for.
            parent_element (Optional[WebElement]): The parent element to search within. If None, searches the entire page.
            max_retries (int): Maximum number of retry attempts.
            retry_delay (int): Delay in seconds between retries.

        Returns:
            Optional[List[WebElement]]: A list of WebElements with the specified class name, or None if not found.
        """
        self.logger.info(f"Retrieving {'sub' if parent_element else ''}elements with class: {class_name}")

        for attempt in range(max_retries):
            try:
                if parent_element:
                    elements = parent_element.find_elements(By.CLASS_NAME, class_name)
                else:
                    elements = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, class_name)))
                
                if not elements:
                    self.logger.warning(f"No {'sub' if parent_element else ''}elements found with class {class_name}")
                    return None
                
                self.logger.debug(f"Found {len(elements)} {'sub' if parent_element else ''}elements with class {class_name}")
                return elements
            
            except (TimeoutException, NoSuchElementException, StaleElementReferenceException):
                self.logger.warning(f"Element not found or stale. Attempt {attempt + 1} of {max_retries}")
            except Exception as e:
                self.logger.error(f"Unexpected error occurred: {str(e)}. Attempt {attempt + 1} of {max_retries}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        self.logger.error(f"Failed to retrieve {'sub' if parent_element else ''}elements with class {class_name} after {max_retries} attempts")
        return None

    def scrape_page_table(self, url: str, table_class: str, pagination_class: str, dropdown_class: str) -> WebElement:
        """
        Scrape a page, handling pagination and returning the table as a DataFrame.

        This method navigates to the specified URL, handles pagination if present,
        extracts the table data, and retrieves links from the table.

        Args:
            url (str): The URL of the page to scrape.
            table_class (str): The class name of the table to scrape.
            pagination_class (str): The class name of the pagination element.
            dropdown_class (str): The class name of the dropdown element for pagination.

        Returns:
            data_table (WebElement): The scraped table as a WebElement.

        """
        self.logger.info(f"Starting page scrape for URL: {url}")
        success = self.go_to_url(url)
        if not success:
            self.logger.warning(f"Page scrape failed: Could not navigate to URL: {url}")
            return None
        self.logger.info("Url found and scraped successfully")
        self.handle_pagination(pagination_class, dropdown_class)
        tables = self.get_elements_by_class(table_class)
        if tables is None:
            self.logger.warning("Page scrape failed: Table not found")
            return None
        
        data_table = tables[0] 

        if data_table is not None:
            self.logger.info("Table found and scraped successfully")
            return data_table
        else:
            self.logger.warning("Page scrape failed: Table not found")
            return None

    def safe_wait_and_click(self, locator: Tuple[str, str], max_attempts: int = 3) -> None:
        """
        Safely wait for an element to be clickable and then click it, with retry logic.

        Args:
            locator (Tuple[str, str]): A tuple of (By.XXX, "value") to locate the element.
            max_attempts (int): Maximum number of attempts to click the element. Defaults to 3.
        
        Raises:
            TimeoutException: If the element is not clickable after max_attempts.
        """
        self.logger.info(f"Attempting to click element: {locator}")
        for attempt in range(max_attempts):
            try:
                element = self.wait.until(EC.element_to_be_clickable(locator))
                element.click()
                self.logger.debug(f"Successfully clicked element on attempt {attempt + 1}")
                return
            except (StaleElementReferenceException, TimeoutException):
                self.logger.warning(f"Failed to click element on attempt {attempt + 1}")
                if attempt == max_attempts - 1:
                    self.logger.error(f"Element {locator} was not clickable after {max_attempts} attempts")
                    raise TimeoutException(f"Element {locator} was not clickable after {max_attempts} attempts")
                self.logger.debug(f"Retrying click attempt in 1 second")
                time.sleep(1)


