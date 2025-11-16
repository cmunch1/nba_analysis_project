#!/bin/bash
set -e

echo "=== Uploading NBA Data to Kaggle ==="

# Get the project root directory (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Create temporary directory for Kaggle upload
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy metadata and data dictionary
cp "$PROJECT_ROOT/data/dataset-metadata.json" $TEMP_DIR/
cp "$PROJECT_ROOT/data/data_dictionary.csv" $TEMP_DIR/ 2>/dev/null || true

# Copy only cumulative_scraped and processed directories
echo "Copying cumulative_scraped directory..."
cp -r "$PROJECT_ROOT/data/cumulative_scraped" $TEMP_DIR/

echo "Copying processed directory (selective files only)..."
mkdir -p $TEMP_DIR/processed
cp "$PROJECT_ROOT/data/processed/column_mapping.json" $TEMP_DIR/processed/ 2>/dev/null || true
cp "$PROJECT_ROOT/data/processed/games_boxscores.csv" $TEMP_DIR/processed/ 2>/dev/null || true
cp "$PROJECT_ROOT/data/processed/teams_boxscores.csv" $TEMP_DIR/processed/ 2>/dev/null || true

# Navigate to temp directory and upload
cd $TEMP_DIR

# Dataset slug from metadata
DATASET_SLUG="chrismunch/nba-game-team-statistics"

echo "Checking if dataset exists..."
if "$PROJECT_ROOT/.venv/bin/kaggle" datasets status $DATASET_SLUG 2>/dev/null; then
    echo "Dataset exists, creating new version..."
    "$PROJECT_ROOT/.venv/bin/kaggle" datasets version -p . -m "Nightly update: $(date +%Y-%m-%d)" -r zip
else
    echo "Dataset doesn't exist, creating initial dataset..."
    "$PROJECT_ROOT/.venv/bin/kaggle" datasets create -p . -r zip
fi

echo "✓ Dataset uploaded successfully!"
echo "View at: https://kaggle.com/datasets/$DATASET_SLUG"
