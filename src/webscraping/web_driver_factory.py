from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from typing import Dict, Any
import logging

class WebDriver(ABC):
    @abstractmethod
    def get(self, url: str):
        pass

    @abstractmethod
    def quit(self):
        pass

class ChromeWebDriver(WebDriver):
    def __init__(self, options: Dict[str, Any] = None):
        chrome_options = webdriver.ChromeOptions()
        if options:
                for option in options:
                    chrome_options.add_argument(option)
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    def get(self, url: str):
        self.driver.get(url)

    def quit(self):
        self.driver.quit()

    def execute_script(self, script, *args):
        return self.driver.execute_script(script, *args)

class WebDriverFactory:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_driver(self, browser: str, options: Dict[str, Any] = None) -> WebDriver:
        browser = browser.lower()
        if browser == "chrome":
            return self._create_chrome_driver(options)
        else:
            self.logger.error(f"Unsupported browser: {browser}")
            raise ValueError(f"Unsupported browser: {browser}")

    def _create_chrome_driver(self, options: Dict[str, Any] = None) -> WebDriver:
        try:
            chrome_options = webdriver.ChromeOptions()
            if options:
                for option in options:
                    chrome_options.add_argument(option)
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            self.logger.info("Chrome WebDriver activated successfully")
            return driver
        except Exception as e:
            self.logger.error(f"Failed to activate Chrome WebDriver: {str(e)}")
            raise
