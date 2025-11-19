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

### 3. Use Container's Pre-installed Packages

**Problem**: `uv run` creates a new virtual environment and downloads all packages (3.5+ GB), causing:
- Slow workflow execution
- "No space left on device" errors
- Defeats the purpose of pre-built Docker containers

**Fix**: Use `python` directly instead of `uv run`

```yaml
# OLD
- name: Run Webscraping
  run: uv run -m src.nba_app.webscraping.main

- name: Run Data Processing
  run: uv run -m src.nba_app.data_processing.main

# NEW
- name: Run Webscraping
  run: python -m src.nba_app.webscraping.main

- name: Run Data Processing
  run: python -m src.nba_app.data_processing.main
```

**Files affected**: Lines 56, 59

---

### 4. Fix Python Module Path in Container

**Problem**: When GitHub Actions checks out code in a container, Python doesn't know where to find project modules, causing `ModuleNotFoundError: No module named 'ml_framework'`.

**Fix**: Set `PYTHONPATH` to the GitHub workspace using the `GITHUB_WORKSPACE` environment variable

```yaml
# OLD
- name: Run Webscraping
  run: python -m src.nba_app.webscraping.main

# NEW
- name: Run Webscraping
  run: |
    export PYTHONPATH="${GITHUB_WORKSPACE}:${PYTHONPATH:-}"
    python -m src.nba_app.webscraping.main
```

**Files affected**: Lines 56-64

**Why this happens**:
- GitHub Actions checks out code to a workspace directory (path varies, e.g., `/__w/nba_analysis_project/nba_analysis_project/`)
- Container's Dockerfile sets `WORKDIR /app` and `PYTHONPATH=/app`
- When checkout happens, code is in the workspace path, not `/app`
- Python needs `PYTHONPATH` set to the actual checkout location to find local modules like `ml_framework`
- `GITHUB_WORKSPACE` is a built-in variable that always points to the correct checkout location

---

## Quick Checklist for Other Workflow Files

When updating other `.yml` workflow files, check for:

- [ ] Update `actions/checkout@v3` → `@v4`
- [ ] Update `actions/upload-artifact@v3` → `@v4`
- [ ] Update any other `@v3` actions to `@v4`
- [ ] Add `options: --user root` to any `container:` blocks
- [ ] Replace `uv run` with `python` for container-based jobs
- [ ] Add `export PYTHONPATH="${GITHUB_WORKSPACE}:${PYTHONPATH:-}"` before Python module execution
- [ ] Test the workflow after changes

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
