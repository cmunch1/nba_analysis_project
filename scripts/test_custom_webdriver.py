#!/usr/bin/env python3
"""Test CustomWebDriver class directly"""

import sys
import os

# Set up Python path
sys.path.insert(0, '/workspace/src')
os.chdir('/workspace')

try:
    print("=== Testing CustomWebDriver ===")
    print()

    # Import the webscraping container
    from nba_app.webscraping.di_container import DIContainer

    # Create container
    print("Creating DIContainer...")
    container = DIContainer()
    container.config.from_yaml('configs/nba/config_path.yaml')
    container.wire(modules=[sys.modules[__name__]])
    print("✓ Container created")
    print()

    # Get the web driver factory
    print("Getting web driver factory...")
    web_driver_factory = container.web_driver_factory()
    print(f"✓ Got web driver factory: {web_driver_factory}")
    print()

    # Create driver
    print("Creating driver...")
    driver = web_driver_factory.create_driver()
    print(f"✓ Driver created successfully: {driver}")
    print()

    # Test navigation
    print("Testing navigation to google.com...")
    driver.get("https://www.google.com")
    print(f"✓ Page title: {driver.title}")
    print()

    # Cleanup
    driver.quit()
    print("✓ Test completed successfully!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
