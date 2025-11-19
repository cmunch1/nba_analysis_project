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

**Problem**: `uv run` (without `--no-sync`) creates a new virtual environment and downloads all packages (3.5+ GB), causing:
- Slow workflow execution
- "No space left on device" errors
- Defeats the purpose of pre-built Docker containers

**Fix**: Use `uv run --no-sync` to use the container's pre-installed packages

```yaml
# OLD (re-installs everything)
- name: Run Webscraping
  run: uv run -m src.nba_app.webscraping.main

# NEW (uses pre-installed packages)
- name: Run Webscraping
  run: uv run --no-sync -m src.nba_app.webscraping.main
```

**Files affected**: Lines 57, 60

**Why this is better than using `python` directly:**
- `uv run --no-sync` automatically handles PYTHONPATH based on `pyproject.toml`
- No need to manually export environment variables
- Consistent with local development workflow
- More explicit about using the project's virtual environment

---

### 4. ~~Fix Python Module Path in Container~~ (NOT NEEDED with `uv run --no-sync`)

**This fix is no longer needed if you use `uv run --no-sync` from fix #3.**

`uv run` automatically handles PYTHONPATH based on your `pyproject.toml` configuration, so you don't need to manually set environment variables.

<details>
<summary>Click here if you're using `python` directly instead of `uv run`</summary>

**Problem**: When GitHub Actions checks out code in a container, Python doesn't know where to find project modules, causing `ModuleNotFoundError: No module named 'ml_framework'`.

**Fix**: Set `PYTHONPATH` to the `src` directory and update module paths

```yaml
# If using python directly (not recommended)
- name: Run Webscraping
  run: |
    export PYTHONPATH="${GITHUB_WORKSPACE}/src:${PYTHONPATH:-}"
    python -m nba_app.webscraping.main
```

**Why this happens**:
- `pyproject.toml` defines packages as `["src/ml_framework", "src/nba_app"]` (line 37)
- This means when installed, `ml_framework` and `nba_app` are top-level packages (not under `src.`)
- Code imports use `from ml_framework.core...` not `from src.ml_framework.core...`
- When running from source (not installed), Python needs `src/` in PYTHONPATH to find these top-level packages
</details>

---

## Quick Checklist for Other Workflow Files

When updating other `.yml` workflow files, check for:

- [ ] Update `actions/checkout@v3` → `@v4`
- [ ] Update `actions/upload-artifact@v3` → `@v4`
- [ ] Update any other `@v3` actions to `@v4`
- [ ] Add `options: --user root` to any `container:` blocks
- [ ] Replace `uv run` with `uv run --no-sync` for container-based jobs
- [ ] Test the workflow after changes

**Note**: If using `uv run --no-sync`, you don't need to manually set PYTHONPATH or remove the `src.` prefix - `uv` handles this automatically!

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
