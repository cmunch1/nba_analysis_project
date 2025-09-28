"""
web_driver.py

This module provides a CustomWebDriver class for managing Selenium WebDriver instances.
It supports multiple browsers and implements custom exception handling and logging.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from typing import Dict, Any, Optional, List
import logging

from .abstract_scraper_classes import AbstractWebDriver
from platform_core.core.config_management.base_config_manager import BaseConfigManager
from platform_core.core.error_handling.error_handler import (
    WebDriverError,
    ConfigurationError
)

logger = logging.getLogger(__name__)

class CustomWebDriver(AbstractWebDriver):
    def __init__(self, config: BaseConfigManager):
        """
        Initialize the WebDriver with configuration.

        Args:
            config (BaseConfigManager): Configuration object.

        Raises:
            ConfigurationError: If there's an issue with the provided configuration.
        """
        try:
            self.config = config
            self.browsers = self.config.browsers if hasattr(self.config, 'browsers') else ['chrome', 'firefox']
            logger.info(f"WebDriver initialized with browsers: {self.browsers}")
        except AttributeError as e:
            raise ConfigurationError(f"Missing required configuration: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error initializing WebDriver: {str(e)}")

    def create_driver(self) -> webdriver.Remote:
        """
        Create and return a WebDriver instance for the first available browser.

        Returns:
            webdriver.Remote: A WebDriver instance.

        Raises:
            WebDriverError: If unable to create a WebDriver for any available browser.
        """
        for browser in self.browsers:
            try:
                driver = self._create_browser_driver(browser)
                if driver:
                    logger.info(f"Successfully created {browser.capitalize()} WebDriver")
                    return driver
            except Exception as e:
                logger.warning(f"Failed to create {browser.capitalize()} WebDriver: {str(e)}")
        
        raise WebDriverError("Failed to create WebDriver with any available browser")

    def _create_browser_driver(self, browser: str) -> webdriver.Remote:
        """
        Create a WebDriver instance for a specific browser.

        Args:
            browser (str): The name of the browser to create a driver for.

        Returns:
            webdriver.Remote: A WebDriver instance.

        Raises:
            WebDriverError: If the specified browser is not supported.
        """
        browser = browser.lower()

        if browser == "chrome":
            return self._create_chrome_driver()
        elif browser == "firefox":
            return self._create_firefox_driver()
        else:
            raise WebDriverError(f"Unsupported browser: {browser}")

    def _create_chrome_driver(self) -> webdriver.Chrome:
        """
        Create a Chrome WebDriver instance.

        Returns:
            webdriver.Chrome: A Chrome WebDriver instance.

        Raises:
            WebDriverError: If there's an error creating the Chrome WebDriver.
        """
        try:
            chrome_options = webdriver.ChromeOptions()
            self._add_browser_options(chrome_options, 'chrome_options')
            return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        except Exception as e:
            raise WebDriverError(f"Error creating Chrome WebDriver: {str(e)}")

    def _create_firefox_driver(self) -> webdriver.Firefox:
        """
        Create a Firefox WebDriver instance.

        Returns:
            webdriver.Firefox: A Firefox WebDriver instance.

        Raises:
            WebDriverError: If there's an error creating the Firefox WebDriver.
        """
        try:
            firefox_options = webdriver.FirefoxOptions()
            self._add_browser_options(firefox_options, 'firefox_options')
            return webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
        except Exception as e:
            raise WebDriverError(f"Error creating Firefox WebDriver: {str(e)}")

    def _add_browser_options(self, options: webdriver.ChromeOptions | webdriver.FirefoxOptions, config_key: str) -> None:
        """
        Add browser-specific options to the WebDriver options.

        Args:
            options: The WebDriver options object.
            config_key (str): The key in the configuration for browser-specific options.

        Raises:
            ConfigurationError: If there's an error processing the browser options.
        """
        try:
            if hasattr(self.config, config_key):
                browser_options = getattr(self.config, config_key)
                if isinstance(browser_options, list):
                    for option in browser_options:
                        options.add_argument(option)
                elif isinstance(browser_options, dict):
                    for key, value in browser_options.items():
                        if value is None:
                            options.add_argument(key)
                        else:
                            options.add_argument(f"{key}={value}")
                logger.debug(f"Added {config_key} to WebDriver options")
            else:
                logger.warning(f"No {config_key} found in configuration")
        except Exception as e:
            raise ConfigurationError(f"Error processing browser options: {str(e)}")

    @staticmethod
    def verify_browser_installations() -> List[str]:
        """
        Verify which browsers are installed and can be used.

        Returns:
            List[str]: A list of installed browsers.
        """
        installed_browsers = []
        try:
            webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
            installed_browsers.append('chrome')
            logger.info("Chrome installation verified")
        except Exception as e:
            logger.warning(f"Chrome installation check failed: {str(e)}")

        try:
            webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))
            installed_browsers.append('firefox')
            logger.info("Firefox installation verified")
        except Exception as e:
            logger.warning(f"Firefox installation check failed: {str(e)}")

        return installed_browsers

    def close_driver(self) -> None:
        """
        Close the WebDriver instance.

        Raises:
            WebDriverError: If there's an error closing the WebDriver.
        """
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                logger.info("WebDriver successfully closed")
        except Exception as e:
            raise WebDriverError(f"Error closing WebDriver: {str(e)}")