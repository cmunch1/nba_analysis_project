#!/bin/bash
# Test Chrome/Chromium in local Docker container

set -e

echo "=========================================="
echo "Testing Chrome/Chromium in Docker Container"
echo "=========================================="
echo ""

# Build the Docker image
echo "Step 1: Building Docker image..."
docker build -t nba-test:local .
echo "✓ Docker image built"
echo ""

# Run the container with interactive shell to debug
echo "Step 2: Running container with Chrome diagnostics..."
echo ""

docker run --rm \
  --user root \
  -v "$(pwd):/workspace" \
  -w /workspace \
  nba-test:local \
  bash -c '
echo "=== System Info ==="
whoami
id
echo ""

echo "=== Chromium Installation ==="
which chromium || echo "chromium not in PATH"
which chromium-browser || echo "chromium-browser not in PATH"
ls -la /usr/bin/chromium* 2>/dev/null || echo "No chromium binaries in /usr/bin"
echo ""

echo "=== ChromeDriver Installation ==="
which chromedriver || echo "chromedriver not in PATH"
ls -la /usr/bin/chromedriver 2>/dev/null || echo "No chromedriver in /usr/bin"
ls -la /usr/lib/chromium/chromedriver 2>/dev/null || echo "No chromedriver in /usr/lib/chromium"
echo ""

echo "=== Versions ==="
chromium --version 2>&1 || echo "Failed to get chromium version"
chromedriver --version 2>&1 || echo "Failed to get chromedriver version"
echo ""

echo "=== Testing Chromium Headless (Direct) ==="
chromium --headless --no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage --disable-gpu --dump-dom https://www.google.com 2>&1 | head -20
echo ""

echo "=== Testing with Python/Selenium ==="
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"
python3 << "PYTHON_EOF"
import sys
import os
sys.path.insert(0, "/workspace/src")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    import shutil

    print("Selenium imported successfully")

    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()

    # Find Chromium binary
    chromium_path = shutil.which("chromium") or "/usr/bin/chromium"
    chrome_options.binary_location = chromium_path
    print(f"Using Chromium at: {chromium_path}")

    # Add minimal required flags
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-setuid-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    # Find ChromeDriver
    chromedriver_path = shutil.which("chromedriver") or "/usr/bin/chromedriver"
    print(f"Using ChromeDriver at: {chromedriver_path}")

    # Create service
    service = ChromeService(executable_path=chromedriver_path)

    # Try to create driver
    print("Creating Chrome WebDriver...")
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
PYTHON_EOF

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
'
