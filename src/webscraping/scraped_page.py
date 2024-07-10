from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import pandas as pd

class ScrapedPage:
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

    def get_table(self, table_class: str) -> pd.DataFrame:
        """Retrieve the table indicated by the given class and return it as a DataFrame."""
        try:
            table = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, table_class)))
            table_html = table.get_attribute('outerHTML')
            dfs = pd.read_html(table_html, header=0)
            return pd.concat(dfs)
        except TimeoutException:
            print("Table not found")
            return pd.DataFrame()

    def get_elements_by_class(self, class_name: str):
        """Retrieve all elements with the given class name."""
        return self.driver.find_elements(By.CLASS_NAME, class_name)

    def parse_ids(self, data_table) -> tuple[pd.Series, pd.Series]:
        """Parse the html table to extract the team and game ids."""
        CLASS_ID = 'Anchor_anchor__cSc3P'
        links = data_table.find_elements(By.CLASS_NAME, CLASS_ID)
        links_list = [i.get_attribute("href") for i in links]
        
        team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
        game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])
        
        return team_id, game_id

    def scrape_page(self, url: str, table_class: str, pagination_class: str, dropdown_class: str) -> pd.DataFrame:
        """
        Scrape a page, handling pagination and returning the table as a DataFrame.
        Also parse and add team and game IDs to the DataFrame.
        """
        self.go_to_url(url)
        self.handle_pagination(pagination_class, dropdown_class)
        df = self.get_table(table_class)
        
        if not df.empty:
            data_table = self.driver.find_element(By.CLASS_NAME, table_class)
            team_id, game_id = self.parse_ids(data_table)
            df['TEAM_ID'] = team_id
            df['GAME_ID'] = game_id
        
        return df