# Kaggle Dataset Setup Guide

This guide covers uploading and maintaining your NBA analysis data on Kaggle as a dataset.

## Overview

Your project currently scrapes NBA data and stores it locally in the `data/` directory. By uploading this data to Kaggle, you can:
- Share your dataset with the community
- Provide a reproducible data source for your project
- Allow others to use your preprocessed NBA data
- Reduce scraping load by using cached data
- Enable the pipeline to run without web scraping dependencies

## Current Data Structure

Your project has the following data directories:
```
data/
├── cumulative_scraped/     # Historical scraped data (all seasons)
│   ├── games_advanced.csv
│   ├── games_four-factors.csv
│   ├── games_misc.csv
│   ├── games_scoring.csv
│   └── games_traditional.csv
├── newly_scraped/          # Latest scraped data (current run)
│   ├── games_advanced.csv
│   ├── games_four-factors.csv
│   ├── games_misc.csv
│   ├── games_scoring.csv
│   ├── games_traditional.csv
│   ├── todays_games_ids.csv
│   └── todays_matchups.csv
├── processed/              # Cleaned and combined data
│   ├── games_boxscores.csv
│   └── teams_boxscores.csv
├── engineered/             # Feature-engineered data
│   └── engineered_features.csv
├── training/               # ML training data
│   ├── training_data.csv
│   └── validation_data.csv
└── dashboard/              # Dashboard-ready data
    ├── dashboard_data.csv
    └── archive/            # Daily snapshots
```

## Step 1: Install Kaggle API

### 1.1 Install the Kaggle Python package

```bash
# Using uv (project's package manager)
uv pip install kaggle

# Or using pip
pip install kaggle
```

### 1.2 Get Kaggle API credentials

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json` file
5. Place it in the correct location:

**Linux/Mac/WSL:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```powershell
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
```

### 1.3 Verify installation

```bash
kaggle datasets list
```

## Step 2: Prepare Dataset for Upload

### 2.1 Create dataset metadata

Create a file `data/dataset-metadata.json`:

```json
{
  "title": "NBA Game Statistics and Analysis Data",
  "id": "YOUR_KAGGLE_USERNAME/nba-game-statistics",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ],
  "keywords": [
    "basketball",
    "nba",
    "sports",
    "machine learning",
    "time series",
    "statistics"
  ],
  "subtitle": "Comprehensive NBA game-level statistics with advanced metrics, four factors, and engineered features for ML prediction",
  "description": "# NBA Game Statistics and Analysis Data\n\nThis dataset contains comprehensive NBA game statistics scraped from NBA.com, processed and engineered for machine learning analysis.\n\n## Data Files\n\n### Raw Scraped Data (cumulative_scraped/)\n- `games_traditional.csv`: Traditional box score stats (points, rebounds, assists, etc.)\n- `games_advanced.csv`: Advanced metrics (offensive/defensive rating, pace, etc.)\n- `games_four-factors.csv`: Four Factors of basketball (shooting, turnovers, rebounding, free throws)\n- `games_misc.csv`: Miscellaneous statistics\n- `games_scoring.csv`: Detailed scoring breakdown\n\n### Processed Data\n- `processed/games_boxscores.csv`: Combined game-level statistics\n- `processed/teams_boxscores.csv`: Team-level aggregated statistics\n- `engineered/engineered_features.csv`: Feature-engineered data with rolling averages, matchup history, etc.\n- `training/training_data.csv`: ML-ready training dataset\n- `training/validation_data.csv`: ML-ready validation dataset\n\n## Use Cases\n- NBA game outcome prediction\n- Player performance analysis\n- Team strength modeling\n- Sports betting analytics\n- Time series forecasting\n\n## Update Frequency\nDataset is updated nightly with the latest NBA game results during the season.\n\n## Source\nData scraped from NBA.com official statistics.\n\n## Related Project\nGitHub: https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project",
  "resources": [
    {
      "path": "cumulative_scraped/games_traditional.csv",
      "description": "Traditional box score statistics for all games"
    },
    {
      "path": "cumulative_scraped/games_advanced.csv",
      "description": "Advanced metrics for all games"
    },
    {
      "path": "cumulative_scraped/games_four-factors.csv",
      "description": "Four Factors statistics for all games"
    },
    {
      "path": "cumulative_scraped/games_misc.csv",
      "description": "Miscellaneous statistics for all games"
    },
    {
      "path": "cumulative_scraped/games_scoring.csv",
      "description": "Scoring breakdown for all games"
    },
    {
      "path": "processed/games_boxscores.csv",
      "description": "Processed and combined game-level statistics"
    },
    {
      "path": "processed/teams_boxscores.csv",
      "description": "Team-level aggregated statistics"
    },
    {
      "path": "engineered/engineered_features.csv",
      "description": "Feature-engineered dataset with rolling averages and matchup history"
    },
    {
      "path": "training/training_data.csv",
      "description": "ML-ready training dataset"
    },
    {
      "path": "training/validation_data.csv",
      "description": "ML-ready validation dataset"
    }
  ]
}
```

**Important:** Replace:
- `YOUR_KAGGLE_USERNAME` with your actual Kaggle username
- `YOUR_GITHUB_USERNAME` with your actual GitHub username

### 2.2 Decide which data to upload

**Option A: Upload all data** (Recommended for full reproducibility)
- Includes raw, processed, engineered, and training data
- Larger dataset (~100-500MB depending on data size)
- Users can skip all preprocessing steps

**Option B: Upload only processed data**
- Includes `processed/`, `engineered/`, and `training/` directories
- Smaller dataset
- Users still need to run initial data processing

**Option C: Upload only cumulative scraped data**
- Includes only `cumulative_scraped/` directory
- Smallest dataset
- Users run full pipeline from data processing onward

## Step 3: Create the Dataset on Kaggle

### 3.1 Initial upload (first time only)

```bash
# Navigate to your data directory
cd /home/chris/projects/nba_analysis_project/data

# Create the dataset on Kaggle
kaggle datasets create -p . -r zip
```

This will:
1. Read the `dataset-metadata.json` file
2. Zip all the CSV files in the directory
3. Upload to Kaggle
4. Create a new dataset at `kaggle.com/datasets/YOUR_USERNAME/nba-game-statistics`

### 3.2 Dataset visibility

After creation, go to your dataset page and:
1. Review the auto-generated description
2. Set visibility to "Public" or "Private"
3. Add a cover image (optional)
4. Add tags for better discoverability

## Step 4: Update Dataset (Nightly Updates)

### 4.1 Create update script

Create `scripts/upload_to_kaggle.sh`:

```bash
#!/bin/bash
set -e

echo "=== Uploading NBA Data to Kaggle ==="

# Navigate to data directory
cd "$(dirname "$0")/../data"

# Update the dataset version
kaggle datasets version -p . -m "Nightly update: $(date +%Y-%m-%d)" -r zip

echo "✓ Dataset updated successfully!"
echo "View at: https://kaggle.com/datasets/YOUR_KAGGLE_USERNAME/nba-game-statistics"
```

Make it executable:
```bash
chmod +x scripts/upload_to_kaggle.sh
```

### 4.2 Add to nightly pipeline

Update your `scripts/run_nightly_pipeline.sh` to include Kaggle upload:

```bash
# At the end of the pipeline, after dashboard prep
echo "Uploading data to Kaggle..."
./scripts/upload_to_kaggle.sh
```

### 4.3 Manual updates

To manually update the dataset:

```bash
cd data
kaggle datasets version -p . -m "Manual update: added new features" -r zip
```

## Step 5: Use Kaggle Data in Your Pipeline

### 5.1 Download data from Kaggle

Create `scripts/download_kaggle_data.sh`:

```bash
#!/bin/bash
set -e

echo "=== Downloading NBA Data from Kaggle ==="

# Create data directory if it doesn't exist
mkdir -p data

# Download and unzip the dataset
kaggle datasets download -d YOUR_KAGGLE_USERNAME/nba-game-statistics -p data --unzip

echo "✓ Data downloaded successfully!"
```

Make it executable:
```bash
chmod +x scripts/download_kaggle_data.sh
```

### 5.2 Update pipeline to support Kaggle mode

Modify `scripts/run_nightly_pipeline.sh` to add a flag:

```bash
#!/bin/bash
set -e

# Parse arguments
USE_KAGGLE=false
SKIP_WEBSCRAPING=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --use-kaggle)
      USE_KAGGLE=true
      shift
      ;;
    --skip-webscraping)
      SKIP_WEBSCRAPING=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Download from Kaggle instead of scraping
if [ "$USE_KAGGLE" = true ]; then
  echo "Using Kaggle dataset instead of web scraping..."
  ./scripts/download_kaggle_data.sh
  SKIP_WEBSCRAPING=true
fi

# Continue with rest of pipeline...
```

### 5.3 Usage examples

```bash
# Normal mode: scrape fresh data
./scripts/run_nightly_pipeline.sh

# Skip web scraping (use existing local data)
./scripts/run_nightly_pipeline.sh --skip-webscraping

# Use Kaggle data instead of scraping
./scripts/run_nightly_pipeline.sh --use-kaggle
```

## Step 6: Benefits and Use Cases

### For Your Project
- **Faster setup**: New contributors can download preprocessed data instead of scraping from scratch
- **GitHub Actions**: Use Kaggle data to avoid scraping in CI/CD (faster, more reliable)
- **Backup**: Your scraped data is safely stored on Kaggle
- **Version control**: Kaggle tracks dataset versions

### For the Community
- **Research**: Others can use your data for NBA analysis
- **Learning**: Data science students can practice ML on real NBA data
- **Reproducibility**: Your published results can be reproduced with the exact data

## Step 7: GitHub Actions Integration

### 7.1 Add Kaggle credentials to GitHub Secrets

1. Go to your GitHub repo settings
2. Navigate to "Secrets and variables" → "Actions"
3. Add two secrets:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: Your API key (from `kaggle.json`)

### 7.2 Update GitHub Actions workflow

Modify `.github/workflows/nightly_pipeline.yml`:

```yaml
env:
  KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
  KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}

jobs:
  run-pipeline:
    steps:
      - name: Set up Kaggle credentials
        run: |
          mkdir -p ~/.kaggle
          echo '{"username":"${{ secrets.KAGGLE_USERNAME }}","key":"${{ secrets.KAGGLE_KEY }}"}' > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json

      - name: Download data from Kaggle
        run: ./scripts/download_kaggle_data.sh

      - name: Run pipeline (skip webscraping)
        run: ./scripts/run_nightly_pipeline.sh --skip-webscraping
```

## Step 8: Best Practices

### Data Size Management
- **Compress large files**: Kaggle handles zip compression automatically
- **Exclude archives**: Don't upload `data/dashboard/archive/` (daily snapshots)
- **File limit**: Kaggle has a 20GB dataset size limit

### Update Frequency
- **Nightly updates**: Update dataset after each successful pipeline run
- **Version messages**: Use descriptive version messages (e.g., "Added 15 new games from 2025-11-15")
- **Major versions**: Create new dataset versions for schema changes

### Documentation
- **README**: Keep the dataset description up-to-date
- **Schema documentation**: Document all columns in each CSV file
- **Change log**: Maintain a version history in the description

### Privacy and Ethics
- **Public data only**: Only upload data from public sources (NBA.com)
- **No personal info**: Ensure no PII or proprietary data
- **Licensing**: Choose appropriate license (CC0-1.0 for public domain)

## Troubleshooting

### "403 Forbidden" error
- Check that `~/.kaggle/kaggle.json` exists and has correct permissions (600)
- Verify your API token is still valid

### "Dataset not found" error
- Ensure the dataset was created successfully
- Check the dataset slug matches `YOUR_USERNAME/nba-game-statistics`

### Upload fails with large files
- Kaggle has a 20GB limit per dataset
- Consider splitting into multiple datasets or excluding archive data

### Version update fails
- Ensure you're in the `data/` directory when running `kaggle datasets version`
- Check that `dataset-metadata.json` exists and is valid JSON

## Next Steps

1. ✅ Install Kaggle API and get credentials
2. ✅ Create `dataset-metadata.json` in `data/` directory
3. ✅ Do initial upload: `kaggle datasets create -p data -r zip`
4. ✅ Create update script: `scripts/upload_to_kaggle.sh`
5. ✅ Create download script: `scripts/download_kaggle_data.sh`
6. ✅ Update pipeline to support `--use-kaggle` flag
7. ✅ Add Kaggle secrets to GitHub Actions
8. ✅ Update GitHub Actions workflow to use Kaggle data

## Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle Datasets Guide](https://www.kaggle.com/docs/datasets)
- [Dataset Metadata Schema](https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata)
