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

from .base_scraper_classes import BaseWebDriver
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.error_handler import (
    WebDriverError,
    ConfigurationError
)

logger = logging.getLogger(__name__)

class CustomWebDriver(BaseWebDriver):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger):
        """
        Initialize the WebDriver with configuration.

        Args:
            config (BaseConfigManager): Configuration object.
            app_logger (BaseAppLogger): Logger instance for error handling.

        Raises:
            ConfigurationError: If there's an issue with the provided configuration.
        """
        try:
            self.config = config
            self.app_logger = app_logger
            self.browsers = self.config.browsers if hasattr(self.config, 'browsers') else ['chrome', 'firefox']
            logger.info(f"WebDriver initialized with browsers: {self.browsers}")
        except AttributeError as e:
            raise ConfigurationError(f"Missing required configuration: {str(e)}", self.app_logger)
        except Exception as e:
            raise ConfigurationError(f"Error initializing WebDriver: {str(e)}", self.app_logger)

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
        
        raise WebDriverError("Failed to create WebDriver with any available browser", self.app_logger)

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
            raise WebDriverError(f"Unsupported browser: {browser}", self.app_logger)

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

            # Set binary location for containerized environments (chromium instead of chrome)
            # This is required for Docker/GitHub Actions where chromium is installed
            import shutil
            import os

            # Try multiple possible chromium locations
            chromium_path = shutil.which('chromium') or shutil.which('chromium-browser') or shutil.which('google-chrome')

            # Also try common installation paths
            if not chromium_path:
                for path in ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']:
                    if os.path.exists(path):
                        chromium_path = path
                        break

            if chromium_path:
                chrome_options.binary_location = chromium_path
                logger.info(f"Using Chromium binary at: {chromium_path}")
                self.app_logger.structured_log(
                    logging.INFO,
                    f"Set Chrome binary location to: {chromium_path}"
                )
            else:
                logger.warning("Could not find Chromium binary, Chrome will use default location")
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Chromium binary not found, using default Chrome location"
                )

            # Prefer a system ChromeDriver when available (better version alignment with chromium package)
            chromedriver_path = shutil.which('chromedriver')

            if not chromedriver_path:
                for path in ['/usr/bin/chromedriver', '/usr/lib/chromium/chromedriver']:
                    if os.path.exists(path):
                        chromedriver_path = path
                        break

            if chromedriver_path:
                logger.info(f"Using system ChromeDriver at: {chromedriver_path}")
                self.app_logger.structured_log(
                    logging.INFO,
                    f"Using system ChromeDriver at: {chromedriver_path}"
                )
                # Enable verbose logging for debugging
                service = ChromeService(
                    executable_path=chromedriver_path,
                    service_args=['--verbose', '--log-path=/tmp/chromedriver.log']
                )
            else:
                logger.warning("System ChromeDriver not found, falling back to webdriver_manager download")
                self.app_logger.structured_log(
                    logging.WARNING,
                    "System ChromeDriver not found, falling back to webdriver_manager download"
                )
                service = ChromeService(
                    ChromeDriverManager().install(),
                    service_args=['--verbose', '--log-path=/tmp/chromedriver.log']
                )

            # CRITICAL: Add headless mode explicitly to prevent X11/display errors
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            logger.info("Added critical headless flags: --headless, --disable-gpu")
            self.app_logger.structured_log(
                logging.INFO,
                "Added critical headless flags to prevent X11/display errors"
            )

            # Check if running as root and enforce critical flags
            if os.getuid() == 0:
                logger.info("Running as root, enforcing --no-sandbox and --disable-setuid-sandbox")
                self.app_logger.structured_log(
                    logging.INFO,
                    "Running as root user, adding required Chrome flags for root execution"
                )
                # These will be added first, before config options, to ensure they're set
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-setuid-sandbox')

            self._add_browser_options(chrome_options, 'chrome_options')

            # Log all final options for debugging
            final_args = chrome_options.arguments if hasattr(chrome_options, 'arguments') else []
            logger.info(f"Final Chrome arguments: {final_args}")
            self.app_logger.structured_log(
                logging.INFO,
                f"Chrome will be launched with {len(final_args)} arguments"
            )
            return webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            raise WebDriverError(f"Error creating Chrome WebDriver: {str(e)}", self.app_logger)

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
            raise WebDriverError(f"Error creating Firefox WebDriver: {str(e)}", self.app_logger)

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
                elif hasattr(browser_options, '__dict__'):
                    # Handle SimpleNamespace objects (from config loading)
                    for key, value in vars(browser_options).items():
                        if value is None:
                            options.add_argument(key)
                        else:
                            options.add_argument(f"{key}={value}")
                logger.debug(f"Added {config_key} to WebDriver options")
            else:
                logger.warning(f"No {config_key} found in configuration")
        except Exception as e:
            raise ConfigurationError(f"Error processing browser options: {str(e)}", self.app_logger)

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
            raise WebDriverError(f"Error closing WebDriver: {str(e)}", self.app_logger)
