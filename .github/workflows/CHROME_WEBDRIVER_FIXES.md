# Chrome/Chromium WebDriver Fixes for GitHub Actions

## Problem

When running Selenium WebDriver with Chrome/Chromium in GitHub Actions containers, you may encounter:

```
Chrome failed to start: exited abnormally.
(unknown error: DevToolsActivePort file doesn't exist)
(The process started from chrome location /usr/bin/chromium is no longer running, so ChromeDriver is assuming that Chrome has crashed.)
```

## Root Causes

1. **Binary Location**: GitHub Actions/Docker containers install `chromium` (not `chrome`), but ChromeDriver looks for `chrome` by default
2. **Version Mismatch (PRIMARY CAUSE)**: Container installs Chromium via `apt install chromium chromium-driver` (Debian's pinned version), but `webdriver_manager` downloads the latest upstream ChromeDriver, causing version incompatibility
3. **Headless Mode Issues**: Chrome needs specific flags to run in containerized/headless environments
4. **Sandbox Issues**: Containers have restricted permissions requiring `--no-sandbox` and `--disable-setuid-sandbox`
5. **Shared Memory**: `/dev/shm` may be too small, requiring `--disable-dev-shm-usage`

## Fixes Applied

### 1. Use System ChromeDriver to Match Container's Chromium Version (CRITICAL FIX)

**File**: `src/nba_app/webscraping/web_driver.py`

```python
def _create_chrome_driver(self) -> webdriver.Chrome:
    try:
        chrome_options = webdriver.ChromeOptions()

        # Set binary location for containerized environments
        import shutil
        import os
        chromium_path = shutil.which('chromium') or shutil.which('chromium-browser') or shutil.which('google-chrome')

        if not chromium_path:
            for path in ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']:
                if os.path.exists(path):
                    chromium_path = path
                    break

        if chromium_path:
            chrome_options.binary_location = chromium_path
            logger.info(f"Using Chromium binary at: {chromium_path}")

        # CRITICAL: Prefer system ChromeDriver to match container's Chromium version
        chromedriver_path = shutil.which('chromedriver')

        if not chromedriver_path:
            for path in ['/usr/bin/chromedriver', '/usr/lib/chromium/chromedriver']:
                if os.path.exists(path):
                    chromedriver_path = path
                    break

        if chromedriver_path:
            logger.info(f"Using system ChromeDriver at: {chromedriver_path}")
            service = ChromeService(executable_path=chromedriver_path)
        else:
            logger.warning("System ChromeDriver not found, falling back to webdriver_manager download")
            service = ChromeService(ChromeDriverManager().install())

        # Check if running as root and enforce critical flags
        if os.getuid() == 0:
            logger.info("Running as root, enforcing --no-sandbox and --disable-setuid-sandbox")
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-setuid-sandbox')

        self._add_browser_options(chrome_options, 'chrome_options')
        return webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        raise WebDriverError(f"Error creating Chrome WebDriver: {str(e)}", self.app_logger)
```

**Why this works**:
- Container installs both Chromium and ChromeDriver from Debian packages (guaranteed version match)
- `shutil.which('chromedriver')` finds the system ChromeDriver that matches Chromium
- Multiple fallback paths for different Chromium installations
- Setting `binary_location` tells ChromeDriver exactly where to find the browser
- `webdriver_manager` is only used as fallback for local development machines
- `os.getuid() == 0` detects if running as root and explicitly adds required flags
- **This is the primary fix that prevents the DevToolsActivePort crash**

### 2. Update Chrome Options in Config

**File**: `configs/nba/webscraping_config.yaml`

Added/updated these critical flags:

```yaml
chrome_options:
  --headless: "new"
  --no-sandbox: null                      # Required for containers (disables sandboxing)
  --disable-dev-shm-usage: null           # Uses /tmp instead of /dev/shm
  --disable-gpu: null                     # Disable GPU hardware acceleration
  --disable-software-rasterizer: null     # Required for GitHub Actions
  --disable-setuid-sandbox: null          # Required for containerized environments
  --window-size=1920,1080: null           # Better than --start-maximized for headless
  --disable-extensions: null
  --disable-popup-blocking: null
  --disable-notifications: null
  --disable-3d-apis: null
  --disable-blink-features=AutomationControlled: null
  --disable-web-security: null
  --ignore-certificate-errors: null
  --ignore-ssl-errors: null
  --user-agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
```

**Key flags for GitHub Actions**:
- `--no-sandbox` - **Critical**: Bypasses OS-level sandboxing (containers already provide isolation)
- `--disable-dev-shm-usage` - **Critical**: Prevents shared memory issues
- `--disable-setuid-sandbox` - **Critical**: Required when running as root or in containers
- `--disable-software-rasterizer` - Prevents software rendering issues
- `--disable-gpu` - GPU not available in containers

### 3. Removed Problematic Options

Removed these flags that can cause issues:
- `--start-maximized` - Replaced with `--window-size=1920,1080` (more reliable in headless)
- `--remote-debugging-port: 9222` - Not needed and can conflict
- `--single-process` - Can cause stability issues
- `--disable-features=VizDisplayCompositor` - Unnecessary and can cause issues

## Testing the Fixes

### Local Testing (with Docker)

```bash
# Build and run container
docker build -t nba-test .
docker run --rm nba-test python -m nba_app.webscraping.main
```

### GitHub Actions Testing

Push changes and manually trigger the workflow:
1. Go to GitHub Actions tab
2. Select "Data Collection (Nightly)" workflow
3. Click "Run workflow"
4. Monitor logs for Chrome initialization

## Expected Success Output

```
INFO:nba_app.webscraping.web_driver:Using Chromium binary at: /usr/bin/chromium
INFO:nba_app.webscraping.web_driver:Successfully created Chrome WebDriver
```

## Fallback Strategy

The code tries Chrome first, then falls back to Firefox if Chrome fails:

```python
browsers:
  - chrome
  - firefox
```

If you still have issues with Chrome, the scraper will automatically try Firefox (though it's not installed in the current Dockerfile).

## Additional Resources

- [ChromeDriver + Docker Guide](https://github.com/SeleniumHQ/docker-selenium)
- [Headless Chrome Flags](https://peter.sh/experiments/chromium-command-line-switches/)
- [Common ChromeDriver Issues](https://stackoverflow.com/questions/50642308/webdriverexception-unknown-error-devtoolsactiveport-file-doesnt-exist-while-t)

---

*Last Updated: 2025-11-19*
