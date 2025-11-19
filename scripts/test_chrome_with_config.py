#!/usr/bin/env python3
"""Test Chrome with exact config options from webscraping_config.yaml"""

import sys
import yaml
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
import shutil

# Load the config
with open('configs/nba/webscraping_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

chrome_options_config = config.get('chrome_options', {})

print("=== Chrome Options from Config ===")
for key, value in chrome_options_config.items():
    print(f"  {key}: {value}")
print()

# Create Chrome options
chrome_options = webdriver.ChromeOptions()

# Set binary location
chromium_path = shutil.which('chromium') or '/usr/bin/chromium'
chrome_options.binary_location = chromium_path
print(f"Using Chromium at: {chromium_path}")

# Add options from config
for key, value in chrome_options_config.items():
    if value is None:
        print(f"Adding flag: {key}")
        chrome_options.add_argument(key)
    else:
        flag = f"{key}={value}"
        print(f"Adding flag: {flag}")
        chrome_options.add_argument(flag)

print()

# Find ChromeDriver
chromedriver_path = shutil.which('chromedriver') or '/usr/bin/chromedriver'
print(f"Using ChromeDriver at: {chromedriver_path}")
print()

# Create service
service = ChromeService(executable_path=chromedriver_path)

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
    import traceback
    traceback.print_exc()
    sys.exit(1)
