# Local Docker Testing Guide

## Purpose

Test your GitHub Actions workflows locally using Docker to debug issues faster without pushing to GitHub.

## Quick Start

```bash
# Test Chrome/Chromium in local Docker container
./scripts/test_chrome_locally.sh
```

This script will:
1. Build the Docker image locally
2. Run comprehensive Chrome/Chromium diagnostics
3. Test Selenium WebDriver with Python
4. Verify navigation works

## Test Results

✅ **All tests pass successfully in local Docker**:
- Chromium 142.0.7444.162 and ChromeDriver 142.0.7444.162 (versions match)
- WebDriver creates successfully
- Navigation to websites works
- All config options are compatible

## Key Finding

**Chrome works perfectly locally but fails in GitHub Actions**, indicating the issue is environment-specific to GitHub Actions, not the code or configuration.

**⚠️ LIKELY CAUSE: Proxy Configuration**

The proxy is configured in GitHub Actions via the `PROXY_URL` secret but NOT in local testing. If the proxy is misconfigured, unreachable, or blocking Chrome, it would cause Chrome to fail in GitHub Actions but work locally.

To test this theory:
```bash
# Test locally WITH a proxy to reproduce the issue
./scripts/test_chrome_with_proxy.sh http://your-proxy:port

# Or disable proxy in GitHub Actions by commenting out in data_collection.yml:
# PROXY_URL: ${{ secrets.PROXY_URL }}
```

## Debugging Tools Created

### 1. `scripts/test_chrome_locally.sh`
Main test script that runs full diagnostic suite in Docker

### 2. `scripts/test_chrome_with_config.py`
Tests Chrome with exact config options from `webscraping_config.yaml`

### 3. `scripts/test_duplicate_flags.py`
Verifies that duplicate Chrome flags don't cause failures

### 4. `scripts/test_chrome_verbose.py`
Tests with verbose ChromeDriver logging enabled

### 5. `scripts/test_chrome_with_proxy.sh`
Tests Chrome with a proxy server configured (to reproduce GitHub Actions environment)
Usage: `./scripts/test_chrome_with_proxy.sh <PROXY_URL>`

## Testing Workflow

```bash
# 1. Make changes to code or config
vim src/nba_app/webscraping/web_driver.py
vim configs/nba/webscraping_config.yaml

# 2. Test locally (no need to push!)
./scripts/test_chrome_locally.sh

# 3. If tests pass locally, commit and push
git add .
git commit -m "Your changes"
git push

# 4. Monitor GitHub Actions for environment-specific issues
```

## Verbose Logging in GitHub Actions

The code now includes verbose ChromeDriver logging:
- Log location: `/tmp/chromedriver.log`
- Enabled via: `service_args=['--verbose', '--log-path=/tmp/chromedriver.log']`
- Workflow displays log automatically after webscraping step

This will show the actual Chrome/Chromium error messages in GitHub Actions.

## Manual Docker Commands

```bash
# Build image
docker build -t nba-test:local .

# Run interactive shell
docker run --rm -it --user root \
  -v $(pwd):/workspace \
  -w /workspace \
  nba-test:local \
  bash

# Inside container, you can:
chromium --version
chromedriver --version
python3 -m nba_app.webscraping.main
```

## Common Issues

### Issue: "the input device is not a TTY"
**Solution**: Remove `-it` flag from docker run command (use `--rm` only)

### Issue: "includes invalid characters for a local volume name"
**Solution**: Use absolute path instead of `$(pwd)`:
```bash
docker run --rm -v /absolute/path:/workspace ...
```

### Issue: D-Bus errors in Chrome output
**Not a problem**: These are harmless warnings in containers without D-Bus daemon

## Next Steps

1. ✅ Chrome works perfectly in local Docker
2. ✅ Verbose logging enabled in code
3. ✅ Workflow will output ChromeDriver log
4. ⏭️ Run GitHub Actions workflow and examine verbose log to identify environment-specific issue

---

*Created: 2025-11-19*
