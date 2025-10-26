# Pipeline Scripts

This directory contains scripts for orchestrating the NBA Analysis Project pipeline.

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
