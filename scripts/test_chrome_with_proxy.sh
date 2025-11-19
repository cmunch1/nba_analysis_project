#!/bin/bash
# Test Chrome with proxy configuration locally

set -e

echo "=========================================="
echo "Testing Chrome with Proxy Configuration"
echo "=========================================="
echo ""

# Check if PROXY_URL argument provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/test_chrome_with_proxy.sh <PROXY_URL>"
    echo "Example: ./scripts/test_chrome_with_proxy.sh http://proxy.example.com:8080"
    exit 1
fi

PROXY_URL="$1"
echo "Testing with proxy: $PROXY_URL"
echo ""

# Build the Docker image
echo "Step 1: Building Docker image..."
docker build -t nba-test:local .
echo "✓ Docker image built"
echo ""

# Run the container with proxy
echo "Step 2: Testing Chrome with proxy..."
echo ""

docker run --rm \
  --user root \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -e PROXY_URL="$PROXY_URL" \
  nba-test:local \
  bash -c '
echo "=== Configure Proxy ==="
/app/scripts/configure_proxy.sh
echo ""

echo "=== Show Modified Config ==="
echo "Chrome options with proxy:"
grep -A5 -B5 "proxy-server" /app/configs/nba/webscraping_config.yaml || echo "Proxy not found in config"
echo ""

echo "=== Test Chrome with Proxy ==="
export PYTHONPATH="/workspace/src"
python3 << "PYTHON_EOF"
import sys
import yaml
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
import shutil

# Load config with proxy
with open("/app/configs/nba/webscraping_config.yaml", "r") as f:
    config = yaml.safe_load(f)

chrome_options_config = config.get("chrome_options", {})

# Create Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = "/usr/bin/chromium"

# Add options from config (including proxy)
for key, value in chrome_options_config.items():
    if value is None:
        chrome_options.add_argument(key)
    else:
        chrome_options.add_argument(f"{key}={value}")
        if "proxy-server" in key:
            print(f"Proxy configured: {value}")

# Add root flags
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-setuid-sandbox")

# Create service
service = ChromeService(
    executable_path="/usr/bin/chromedriver",
    service_args=["--verbose", "--log-path=/tmp/chromedriver.log"]
)

# Try to create driver
print("Creating Chrome WebDriver with proxy...")
try:
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("✓ WebDriver created successfully with proxy!")

    # Test navigation
    print("Testing navigation to google.com through proxy...")
    driver.get("https://www.google.com")
    print(f"✓ Page title: {driver.title}")

    # Cleanup
    driver.quit()
    print("✓ Test completed successfully with proxy!")

except Exception as e:
    print(f"✗ Error with proxy: {e}")
    print()
    print("=== ChromeDriver Log ===")
    import os
    if os.path.exists("/tmp/chromedriver.log"):
        with open("/tmp/chromedriver.log", "r") as f:
            print(f.read())
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF
'

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
