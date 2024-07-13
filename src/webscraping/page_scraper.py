from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.remote.webelement import WebElement
import time
import pandas as pd

class PageScraper:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
        self.wait = WebDriverWait(self.driver, 10)

    def go_to_url(self, url: str):
        """Navigate to the given URL."""
        self.driver.get(url)

    def handle_pagination(self, pagination_class: str, dropdown_class: str):
        """Handle pagination by selecting 'ALL' in the dropdown if it exists."""
        try:
            pagination = self.driver.find_element(By.CLASS_NAME, pagination_class)
            try:
                page_dropdown = pagination.find_element(By.CLASS_NAME, dropdown_class)
                page_dropdown.send_keys("ALL")
                time.sleep(3)
                self.driver.execute_script('arguments[0].click()', page_dropdown)
                time.sleep(3)
            except NoSuchElementException:
                print("No dropdown found")
        except NoSuchElementException:
            print("No pagination found")

    def get_table(self, table_class: str) -> WebElement:
        """Retrieve the table indicated by the given class."""
        try:
            data_table = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, table_class)))
            return data_table
        except TimeoutException:
            print("Table not found")
            return None
        
    def convert_table_df(self, data_table: WebElement) -> pd.DataFrame:
        """Convert the html table to a DataFrame."""
        table_html = data_table.get_attribute('outerHTML')
        df = pd.read_html(table_html, header=0)
        return pd.concat(df)
        
    def get_table_links(self, data_table: WebElement) -> list[str]:
        """Parse the table to get any links"""
        links = data_table.find_elements(By.CLASS_NAME, CLASS_ID)
        return links
    
    def extract_hrefs(self, links: WebElement) -> list[str]:
        """Extract the hrefs from a list of links."""
        hrefs = [i.get_attribute("href") for i in links]
        return hrefs

    def get_elements_by_class(self, class_name: str) -> list[WebElement]:
        """Retrieve all elements with the given class name."""
        return self.driver.find_elements(By.CLASS_NAME, class_name)

    def scrape_page_table(self, url: str, table_class: str, pagination_class: str, dropdown_class: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """
        Scrape a page, handling pagination and returning the table as a DataFrame.

        """
        self.go_to_url(url)
        self.handle_pagination(pagination_class, dropdown_class)
        data_table = self.get_table(table_class)
        
        if data_table is not None:
            df = self.convert_table_df(data_table)
            links = self.get_table_links(data_table)
            hrefs = self.extract_hrefs(links)
        else:
            df = None
            links = None
            hrefs = None

        return df, links, hrefs
        
