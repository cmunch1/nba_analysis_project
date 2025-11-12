# Pipeline Scripts

This directory contains helper scripts for the NBA Analysis Project.

## Overview

| Script | Purpose | Requires Secrets |
|--------|---------|------------------|
| `run_nightly_pipeline.sh` | Main pipeline orchestration | No (optional proxy) |
| `download_kaggle_data.sh` | Download public Kaggle datasets | No |
| `setup_fork.sh` | Interactive setup wizard for forkers | No |
| `detect_gpu.sh` | Detect GPU and recommend Docker setup | No |

## Quick Start

```bash
# For new users - setup wizard
./scripts/setup_fork.sh

# Check GPU availability (for Docker)
./scripts/detect_gpu.sh

# Download data only
./scripts/download_kaggle_data.sh

# Run ML pipeline with Kaggle data
./scripts/run_nightly_pipeline.sh --data-source kaggle
```

---

## run_nightly_pipeline.sh

Main orchestration script that runs the complete batch pipeline for NBA win prediction.

### Pipeline Stages

The script executes the following stages in order:

1. **Webscraping** - Scrape today's game schedule and yesterday's results
2. **Data Processing** - Clean and consolidate scraped data
3. **Feature Engineering** - Generate 849 features with deterministic ordering
4. **Inference** - Generate predictions with uncertainty quantification
5. **Dashboard Prep** - Aggregate predictions and results (optional)

### Usage

```bash
# Run full pipeline (skip webscraping and dashboard by default)
./scripts/run_nightly_pipeline.sh

# Run without webscraping (use existing scraped data)
./scripts/run_nightly_pipeline.sh --skip-webscraping

# Skip dashboard prep (default due to known blocker)
./scripts/run_nightly_pipeline.sh --skip-dashboard

# Include dashboard prep (not recommended - has known data schema issue)
./scripts/run_nightly_pipeline.sh --include-dashboard

# Show help
./scripts/run_nightly_pipeline.sh --help
```

### Options

| Option | Description |
|--------|-------------|
| `--skip-webscraping` | Skip stage 1 (use existing scraped data) |
| `--skip-dashboard` | Skip stage 5 (default - dashboard prep has known blocker) |
| `--include-dashboard` | Force include dashboard prep stage |
| `-h`, `--help` | Show usage information |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - all stages completed |
| 1 | Stage 1 failed (webscraping) |
| 2 | Stage 2 failed (data processing) |
| 3 | Stage 3 failed (feature engineering) |
| 4 | Stage 4 failed (inference) |
| 5 | Stage 5 failed (dashboard prep) |
| 99 | Setup/configuration error |

### Environment Variables

The script respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URI | `file:///<project_dir>/mlruns` |
| `PROXY_URL` | Proxy URL for webscraping | None |

### Output

The script generates:

- **Log file**: `logs/pipeline_<timestamp>.log` - Detailed execution log
- **Data files**:
  - `data/newly_scraped/todays_matchups.csv` (if webscraping enabled)
  - `data/processed/teams_boxscores.csv`
  - `data/engineered/engineered_features.csv`
  - `data/predictions/predictions_<date>.csv`
  - `data/dashboard/dashboard_data.csv` (if dashboard enabled)

### Example Run

```bash
$ ./scripts/run_nightly_pipeline.sh --skip-webscraping

[INFO] NBA Nightly Pipeline Starting
[INFO] Timestamp: 2025-10-26 13:57:11
[INFO] Project Directory: /home/chris/projects/nba_analysis_project
[INFO] Log File: logs/pipeline_20251026_135711.log
[WARNING] MLFLOW_TRACKING_URI not set, using local mlruns
[INFO] Configuration:
[INFO]   Skip Webscraping: true
[INFO]   Skip Dashboard: true
[WARNING] Skipping Stage 1 (Webscraping) - using existing data

================================================================================
  STAGE 2: Data Processing (Consolidation & Cleaning)
================================================================================
[INFO] Running: uv run -m src.nba_app.data_processing.main
[SUCCESS] Stage 2 (Data Processing) completed in 6s

================================================================================
  STAGE 3: Feature Engineering (849 Features with Deterministic Ordering)
================================================================================
[INFO] Running: uv run -m src.nba_app.feature_engineering.main
[SUCCESS] Stage 3 (Feature Engineering) completed in 57s

================================================================================
  STAGE 4: Inference (Predictions with Uncertainty Quantification)
================================================================================
[INFO] Running: uv run -m src.nba_app.inference.main
[SUCCESS] Stage 4 (Inference) completed in 5s
[WARNING] Skipping Stage 5 (Dashboard Prep) - known data schema blocker

================================================================================
  PIPELINE COMPLETE
================================================================================
[SUCCESS] Pipeline completed successfully in 68s
[INFO] Timestamp: 2025-10-26 13:58:19
[INFO] Output Files:
[SUCCESS]   ✓ data/processed/teams_boxscores.csv
[SUCCESS]   ✓ data/engineered/engineered_features.csv
[SUCCESS]   ✓ data/predictions/predictions_2025-10-26.csv
[INFO]     Generated 12 predictions
[INFO] Full log available at: logs/pipeline_20251026_135711.log
[SUCCESS] All done! 🏀
```

### Features

- ✅ **Colored output** - Easy to scan console output
- ✅ **Detailed logging** - All output captured to timestamped log files
- ✅ **Error handling** - Exits on first failure with clear error codes
- ✅ **Performance tracking** - Reports execution time for each stage
- ✅ **Output validation** - Verifies expected files were created
- ✅ **Prediction summary** - Shows count of predictions generated
- ✅ **Flexible execution** - Skip stages as needed with command-line flags

### Known Issues

1. **Dashboard Prep Blocker**
   - Stage 5 (Dashboard Prep) has a known data schema issue
   - Missing `is_home_team` column in processed data
   - Currently skipped by default (`--skip-dashboard`)
   - See [DEPLOYMENT_PLAN.md](../DEPLOYMENT_PLAN.md) Phase 2 for details

### Integration with Cron/Scheduler

For nightly execution (3am EST):

```bash
# Add to crontab (crontab -e)
0 3 * * * cd /path/to/nba_analysis_project && ./scripts/run_nightly_pipeline.sh >> logs/cron.log 2>&1
```

For GitHub Actions:

```yaml
name: Nightly Pipeline
on:
  schedule:
    - cron: '0 8 * * *'  # 3am EST = 8am UTC
  workflow_dispatch:
jobs:
  run_pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Pipeline
        run: ./scripts/run_nightly_pipeline.sh --skip-webscraping
```

### Troubleshooting

**Problem**: Script won't execute
**Solution**: Ensure it's executable: `chmod +x scripts/run_nightly_pipeline.sh`

**Problem**: "uv command not found"
**Solution**: Install uv: `pip install uv`

**Problem**: Stage fails with MLflow error
**Solution**: Set `MLFLOW_TRACKING_URI` environment variable or use local default

**Problem**: Windows line ending errors
**Solution**: Convert to Unix format: `sed -i 's/\r$//' scripts/run_nightly_pipeline.sh`

### Development

To modify the pipeline:

1. Edit `scripts/run_nightly_pipeline.sh`
2. Test changes: `./scripts/run_nightly_pipeline.sh --skip-webscraping`
3. Review log file in `logs/pipeline_*.log`
4. Commit changes

### Related Documentation

- [DEPLOYMENT_PLAN.md](../DEPLOYMENT_PLAN.md) - Full deployment roadmap
- [configs/nba/](../configs/nba/) - Configuration files for each module
- [src/nba_app/](../src/nba_app/) - Source code for pipeline modules

---

## download_kaggle_data.sh

Simple script to download public NBA datasets from Kaggle. No authentication required for public datasets.

### Usage

```bash
# Download default datasets
./scripts/download_kaggle_data.sh

# Download custom dataset
./scripts/download_kaggle_data.sh --dataset username/dataset-name

# Download custom processed data
./scripts/download_kaggle_data.sh --processed username/processed-dataset

# Show help
./scripts/download_kaggle_data.sh --help
```

### Options

| Option | Description |
|--------|-------------|
| `--dataset USERNAME/DATASET` | Specify custom cumulative scraped data dataset |
| `--processed USERNAME/DATASET` | Specify custom processed data dataset |
| `-h`, `--help` | Show usage information |

### What It Downloads

1. **Cumulative Scraped Data** (~23 MB)
   - `games_traditional.csv`
   - `games_advanced.csv`
   - `games_four-factors.csv`
   - `games_misc.csv`
   - `games_scoring.csv`

2. **Processed Data** (~10 MB)
   - `teams_boxscores.csv`
   - `games_boxscores.csv`
   - `column_mapping.json`

### Example

```bash
$ ./scripts/download_kaggle_data.sh
Downloading NBA data from Kaggle...

Downloading cumulative scraped data...
Dataset: YOUR_KAGGLE_USERNAME/nba-game-stats-daily
✓ Cumulative scraped data downloaded
Files:
  data/cumulative_scraped/games_traditional.csv (15M)
  data/cumulative_scraped/games_advanced.csv (12M)
  ...

Downloading processed data...
Dataset: YOUR_KAGGLE_USERNAME/nba-processed-data
✓ Processed data downloaded
Files:
  data/processed/teams_boxscores.csv (8.2M)
  ...

✓ Data download complete!

Next steps:
  1. Run ML pipeline: ./scripts/run_nightly_pipeline.sh --data-source kaggle
  2. View dashboard: streamlit run streamlit_app/app.py
```

### Troubleshooting

**Problem**: "Dataset not found"
**Solution**: Verify dataset exists and is public at https://kaggle.com/datasets/YOUR_USERNAME/dataset-name

**Problem**: Kaggle CLI not installed
**Solution**: Script auto-installs with `pip install kaggle`

---

## setup_fork.sh

Interactive setup wizard for people forking the repository. Guides through dependency installation and data source selection.

### Usage

```bash
./scripts/setup_fork.sh
```

### What It Does

1. **Checks Environment**
   - Detects if repo is a fork
   - Verifies project directory

2. **Installs Dependencies**
   - Installs `uv` if not present
   - Runs `uv sync` to install all packages

3. **Data Source Selection**
   - **Option 1 (Kaggle)**: Downloads public datasets automatically
   - **Option 2 (Local)**: Uses data already in repository
   - **Option 3 (Scrape)**: Guides through proxy setup for scraping

4. **Optional Testing**
   - Offers to run quick pipeline test
   - Validates setup is working

### Interactive Flow

```bash
$ ./scripts/setup_fork.sh

============================================
  NBA Prediction Project - Setup Wizard
============================================

✓ Fork detected - great!

Step 1: Installing Dependencies
...
✓ Dependencies installed

Step 2: Choose Data Source

You have three options:
  1) Kaggle - Download public datasets (recommended for getting started)
  2) Local - Use data already in the repository (if committed)
  3) Scrape - Scrape fresh data yourself (requires proxy)

Enter choice [1-3]: 1

Setting up with Kaggle data...
✓ Data downloaded successfully!

Step 3: Testing Setup

Would you like to run a quick test of the ML pipeline?
Run test? [y/N]: y

Running ML pipeline test (this may take a few minutes)...
✓ Test completed successfully!

============================================
  Setup Complete!
============================================

Next steps:

  Run ML Pipeline:
    ./scripts/run_nightly_pipeline.sh --data-source kaggle

  View Dashboard:
    streamlit run streamlit_app/app.py

  Run with Docker:
    docker-compose up nba-pipeline

  Test GitHub Actions:
    Go to Actions → 'Local Development' → Run workflow

Happy coding! 🏀
```

### For Different User Types

**New Users / Contributors:**
- Choose Option 1 (Kaggle)
- No secrets required
- Start experimenting immediately

**Experienced Users:**
- Choose Option 2 (Local) if you have data
- Choose Option 3 (Scrape) if you want fresh data and have proxy

**Project Maintainer:**
- Choose Option 3 (Scrape)
- Set `PROXY_URL` environment variable
- Run full pipeline with scraping

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Setup completed successfully |
| 1 | Setup failed (error during installation) |

---

## detect_gpu.sh

Auto-detection script for GPU availability and Docker GPU support. Helps users determine if they should use CPU or GPU Docker images.

### Usage

```bash
./scripts/detect_gpu.sh
```

### What It Checks

1. **NVIDIA GPU Presence**
   - Runs `nvidia-smi` to detect GPU
   - Shows GPU model, driver version, memory

2. **nvidia-docker Runtime**
   - Checks if Docker can access GPU
   - Validates nvidia-docker2 installation

3. **Recommendations**
   - Suggests appropriate docker-compose command
   - Provides setup instructions if GPU found but docker not configured

### Example Output

**With GPU and nvidia-docker:**
```bash
$ ./scripts/detect_gpu.sh

=== GPU Detection ===

✓ NVIDIA GPU detected

NVIDIA GeForce RTX 3090, 525.147.05, 24576 MiB

✓ nvidia-docker runtime is available

Recommended setup:
  docker-compose -f docker-compose.gpu.yml up

Or build GPU image:
  docker build -f Dockerfile.gpu -t nba-pipeline:gpu .
```

**With GPU but no nvidia-docker:**
```bash
$ ./scripts/detect_gpu.sh

=== GPU Detection ===

✓ NVIDIA GPU detected

NVIDIA GeForce RTX 3090, 525.147.05, 24576 MiB

⚠ nvidia-docker runtime not detected

To enable GPU support in Docker:
  1. Install nvidia-docker2:
     [installation commands...]

  2. Test GPU access:
     docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

For now, use CPU version:
  docker-compose up
```

**No GPU:**
```bash
$ ./scripts/detect_gpu.sh

=== GPU Detection ===

✗ No NVIDIA GPU detected

Running on CPU. Use standard Docker setup:
  docker-compose up
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | GPU detected and nvidia-docker available |
| 1 | GPU detected but nvidia-docker not available |
| 2 | No GPU detected |

### Integration with setup_fork.sh

The setup wizard automatically runs this script and:
- Uses GPU docker-compose if available
- Falls back to CPU if not
- Provides instructions for enabling GPU support

### Troubleshooting

**Problem**: Script says no GPU but you have one
**Solution**: Install nvidia-utils: `sudo apt-get install nvidia-utils-525` (or your driver version)

**Problem**: GPU detected but docker can't access it
**Solution**: Follow nvidia-docker2 installation instructions in script output

---

## Additional Scripts (Future)

Planned helper scripts:

- `upload_to_kaggle.sh` - Upload local data to Kaggle (maintainer only)
- `clean_old_predictions.sh` - Archive old prediction files
- `validate_data.sh` - Run data validation checks
- `benchmark_pipeline.sh` - Performance benchmarking

---

## Environment Variables Reference

All scripts respect these environment variables:

| Variable | Used By | Purpose | Required |
|----------|---------|---------|----------|
| `MLFLOW_TRACKING_URI` | `run_nightly_pipeline.sh` | MLflow server location | No (defaults to local) |
| `PROXY_URL` | `run_nightly_pipeline.sh` | Proxy for NBA.com scraping | Only for scraping |
| `KAGGLE_USERNAME` | Kaggle upload scripts | Kaggle account username | Only for uploads |
| `KAGGLE_KEY` | Kaggle upload scripts | Kaggle API key | Only for uploads |

---

## CI/CD Integration

These scripts are used by GitHub Actions workflows:

- **data_collection.yml**: Uses `run_nightly_pipeline.sh --data-source scrape`
- **ml_pipeline.yml**: Downloads via Kaggle CLI, then runs pipeline
- **local_dev.yml**: Uses `run_nightly_pipeline.sh` with user choice

See [../.github/workflows/](../.github/workflows/) for workflow definitions.
