#!/usr/bin/env python3
"""Test Chrome with verbose logging to see actual error"""

import sys
import yaml
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
import shutil
import os

# Load the config
with open('configs/nba/webscraping_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

chrome_options_config = config.get('chrome_options', {})

# Create Chrome options
chrome_options = webdriver.ChromeOptions()

# Set binary location
chromium_path = shutil.which('chromium') or '/usr/bin/chromium'
chrome_options.binary_location = chromium_path
print(f"Using Chromium at: {chromium_path}")

# Add root flags first (as code does)
if os.getuid() == 0:
    print("Running as root, adding root flags first")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-setuid-sandbox')

# Add options from config
for key, value in chrome_options_config.items():
    if value is None:
        chrome_options.add_argument(key)
    else:
        chrome_options.add_argument(f"{key}={value}")

print()

# Find ChromeDriver
chromedriver_path = shutil.which('chromedriver') or '/usr/bin/chromedriver'
print(f"Using ChromeDriver at: {chromedriver_path}")
print()

# Create service with VERBOSE logging
print("Creating ChromeDriver service with verbose logging...")
service = ChromeService(
    executable_path=chromedriver_path,
    log_output='chromedriver.log',
    service_args=['--verbose']
)

# Try to create driver
print("Creating Chrome WebDriver...")
try:
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("✓ WebDriver created successfully!")

    # Test navigation
    print("Testing navigation to google.com...")
    driver.get("https://www.google.com")
    print(f"✓ Page title: {driver.title}")

    # Cleanup
    driver.quit()
    print("✓ Test completed successfully!")

except Exception as e:
    print(f"✗ Error: {e}")
    print()
    print("=== ChromeDriver Log ===")
    if os.path.exists('chromedriver.log'):
        with open('chromedriver.log', 'r') as f:
            print(f.read())
    import traceback
    traceback.print_exc()
    sys.exit(1)
