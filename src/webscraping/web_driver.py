
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from typing import Dict, Any
import logging

from .abstract_scraper_classes import (
    AbstractWebDriver,
)
from ..config.config import AbstractConfig

class WebDriver_(AbstractWebDriver):
    def __init__(self, config: AbstractConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_driver(self, browser: str = 'chrome'):
        options = self.config.webdriver_options
        browser = browser.lower()
        if browser == "chrome":
            return self._create_chrome_driver(options)
        else:
            self.logger.error(f"Unsupported browser: {browser}")
            raise ValueError(f"Unsupported browser: {browser}")

    def _create_chrome_driver(self, options: Dict[str, Any] = None):
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
