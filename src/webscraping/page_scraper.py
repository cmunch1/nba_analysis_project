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

from ..config.config import config

class PageScraper:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
        self.wait = WebDriverWait(self.driver, config.wait_time)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info("PageScraper initialized")

    def go_to_url(self, url: str) -> bool:
        self.logger.info(f"Navigating to URL: {url}")
        try:
            self.driver.get(url)
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
                self.wait.until(EC.staleness_of(pagination))
                self.logger.debug("Original pagination element became stale")
                self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, pagination_class)))
                self.logger.debug("New pagination element loaded")
            except NoSuchElementException:
                self.logger.warning("No dropdown found in pagination element")
        except TimeoutException:
            self.logger.warning("No pagination found on the page")

    def get_elements_by_class(self, class_name: str, parent_element: Optional[WebElement] = None) -> Optional[List[WebElement]]:
        self.logger.info(f"Retrieving {'sub' if parent_element else ''}elements with class: {class_name}")
        for attempt in range(config.max_retries):
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
                self.logger.warning(f"Element not found or stale. Attempt {attempt + 1} of {config.max_retries}")
            except Exception as e:
                self.logger.error(f"Unexpected error occurred: {str(e)}. Attempt {attempt + 1} of {config.max_retries}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay)
        self.logger.error(f"Failed to retrieve {'sub' if parent_element else ''}elements with class {class_name} after {config.max_retries} attempts")
        return None

    def scrape_page_table(self, url: str, table_class: str, pagination_class: str, dropdown_class: str) -> WebElement:
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

    def safe_wait_and_click(self, locator: Tuple[str, str]) -> None:
        self.logger.info(f"Attempting to click element: {locator}")
        for attempt in range(config.max_retries):
            try:
                element = self.wait.until(EC.element_to_be_clickable(locator))
                element.click()
                self.logger.debug(f"Successfully clicked element on attempt {attempt + 1}")
                return
            except (StaleElementReferenceException, TimeoutException):
                self.logger.warning(f"Failed to click element on attempt {attempt + 1}")
                if attempt == config.max_retries - 1:
                    self.logger.error(f"Element {locator} was not clickable after {config.max_retries} attempts")
                    raise TimeoutException(f"Element {locator} was not clickable after {config.max_retries} attempts")
                self.logger.debug(f"Retrying click attempt in {config.retry_delay} seconds")
                time.sleep(config.retry_delay)



