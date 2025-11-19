#!/usr/bin/env python3
"""Test if duplicate Chrome flags cause issues"""

import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
import shutil

# Create Chrome options
chrome_options = webdriver.ChromeOptions()

# Set binary location
chromium_path = '/usr/bin/chromium'
chrome_options.binary_location = chromium_path
print(f"Using Chromium at: {chromium_path}")

# Add --no-sandbox twice (simulating root detection + config)
print("Adding --no-sandbox (first time)")
chrome_options.add_argument('--no-sandbox')
print("Adding --disable-setuid-sandbox (first time)")
chrome_options.add_argument('--disable-setuid-sandbox')

# Add other options
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')

# Add --no-sandbox again (second time from config)
print("Adding --no-sandbox (second time)")
chrome_options.add_argument('--no-sandbox')
print("Adding --disable-setuid-sandbox (second time)")
chrome_options.add_argument('--disable-setuid-sandbox')

chrome_options.add_argument('--window-size=1920,1080')

print()

# Find ChromeDriver
chromedriver_path = '/usr/bin/chromedriver'
print(f"Using ChromeDriver at: {chromedriver_path}")
print()

# Create service
service = ChromeService(executable_path=chromedriver_path)

# Try to create driver
print("Creating Chrome WebDriver with duplicate flags...")
try:
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("✓ WebDriver created successfully!")

    # Test navigation
    print("Testing navigation to google.com...")
    driver.get("https://www.google.com")
    print(f"✓ Page title: {driver.title}")

    # Cleanup
    driver.quit()
    print("✓ Test completed - duplicate flags are OK!")

except Exception as e:
    print(f"✗ Error with duplicate flags: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
