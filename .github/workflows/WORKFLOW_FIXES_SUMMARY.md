# GitHub Actions Workflow Fixes Summary

This document summarizes the changes made to `data_collection.yml` to fix common issues. Apply these same fixes to other workflow files as needed.

## Changes Made

### 1. Update Deprecated Actions

**Problem**: GitHub Actions v3 actions are deprecated and will fail.

**Fix**: Update to v4 versions

```yaml
# OLD
uses: actions/checkout@v3
uses: actions/upload-artifact@v3

# NEW
uses: actions/checkout@v4
uses: actions/upload-artifact@v4
```

**Files affected**: Lines 30, 66

---

### 2. Fix Container Permission Issues

**Problem**: Container user lacks permissions to write to GitHub Actions temp directories, causing `EACCES: permission denied` errors.

**Fix**: Add `options: --user root` to container configuration

```yaml
container:
  image: ghcr.io/${{ github.repository }}:latest
  options: --user root  # <- ADD THIS LINE
  credentials:
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}
```

**Files affected**: Line 21

---

### 3. Use Container's Pre-installed Python Instead of `uv run`

**Problem**: `uv run` creates a new virtual environment and downloads all packages (3.5+ GB), causing:
- Slow workflow execution
- "No space left on device" errors
- Defeats the purpose of pre-built Docker containers

**Even `uv run --no-sync` doesn't work** because:
- The container's venv is at `/app/.venv`
- GitHub Actions checks out code to a different location
- `uv` creates a new empty `.venv` in the workspace
- The new venv has no packages installed

**Fix**: Use `python` directly from the container's PATH

```yaml
# OLD (creates new venv and re-installs everything)
- name: Run Webscraping
  run: uv run -m src.nba_app.webscraping.main

# NEW (uses container's pre-installed Python)
- name: Run Webscraping
  run: |
    export PYTHONPATH="${GITHUB_WORKSPACE}/src:${PYTHONPATH:-}"
    python -m nba_app.webscraping.main
```

**Files affected**: Lines 56-64

**Why this works:**
- Container's Dockerfile sets `ENV PATH="/app/.venv/bin:$PATH"` (line 71)
- When you run `python`, it uses the container's pre-installed Python from `/app/.venv/bin/python`
- All packages are already installed in that venv
- We just need to set `PYTHONPATH` so Python can find the project modules

---

### 4. Set PYTHONPATH for Python Module Resolution

**Problem**: When GitHub Actions checks out code in a container, Python doesn't know where to find project modules, causing `ModuleNotFoundError: No module named 'ml_framework'`.

**Fix**: Set `PYTHONPATH` to the `src` directory (already shown in Fix #3)

**Why this is needed**:
- `pyproject.toml` defines packages as `["src/ml_framework", "src/nba_app"]` (line 37)
- This means when installed, `ml_framework` and `nba_app` are top-level packages (not under `src.`)
- Code imports use `from ml_framework.core...` not `from src.ml_framework.core...`
- When running from source (not installed), Python needs `src/` in PYTHONPATH to find these top-level packages
- We also remove the `src.` prefix from module names (use `nba_app.webscraping.main` not `src.nba_app.webscraping.main`)

---

### 5. Fix ChromeDriver Version Mismatch

**Problem**: Selenium WebDriver crashes with `DevToolsActivePort file doesn't exist` because:
- GitHub Actions container installs Chromium via `apt install chromium chromium-driver` (Debian's pinned version)
- `webdriver_manager` downloads the latest upstream ChromeDriver
- Version mismatch between container's Chromium and downloaded ChromeDriver causes crashes

**Fix**: Prefer system-installed ChromeDriver from the container

**File**: `src/nba_app/webscraping/web_driver.py`

```python
# After locating Chromium binary, detect system ChromeDriver
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

return webdriver.Chrome(service=service, options=chrome_options)
```

**Why this works**:
- Container's Chromium and ChromeDriver are installed together from the same Debian package
- Versions are guaranteed to be compatible
- `webdriver_manager` is only used as fallback for local development
- Eliminates version mismatch crashes in GitHub Actions

**Debugging Step**: Add ChromeDriver version check to workflow

```yaml
- name: Debug Chromium Installation
  run: |
    echo "=== Chromedriver Version ==="
    which chromedriver || echo "chromedriver not in PATH"
    chromedriver --version || echo "Failed to get chromedriver version"
```

**Files affected**:
- `src/nba_app/webscraping/web_driver.py` (lines 133-155)
- `.github/workflows/data_collection.yml` (lines 66-68)

---

## Quick Checklist for Other Workflow Files

When updating other `.yml` workflow files, check for:

- [ ] Update `actions/checkout@v3` → `@v4`
- [ ] Update `actions/upload-artifact@v3` → `@v4`
- [ ] Update any other `@v3` actions to `@v4`
- [ ] Add `options: --user root` to any `container:` blocks
- [ ] Replace `uv run` with `python` for container-based jobs
- [ ] Add `export PYTHONPATH="${GITHUB_WORKSPACE}/src:${PYTHONPATH:-}"` before Python commands
- [ ] Remove `src.` prefix from module names (use `nba_app.` not `src.nba_app.`)
- [ ] Test the workflow after changes
- [ ] Prefer system-installed Chromium/ChromeDriver in Selenium config to avoid version mismatch
- [ ] Log `chromedriver --version` in debugging steps to confirm driver availability

---

## When NOT to Apply These Fixes

- **Non-container jobs**: If the job doesn't use `container:`, skip fix #2 and #3
- **Non-UV projects**: If not using `uv`, skip fix #3
- **Custom permissions needed**: If the container needs specific non-root permissions, adjust fix #2 accordingly

---

## Testing

After applying these fixes, test by:
1. Pushing changes to GitHub
2. Manually triggering the workflow via `workflow_dispatch`
3. Monitoring for permission errors or package installation issues

---

*Generated: 2025-11-19*
*Original file: `.github/workflows/data_collection.yml`*
