#!/bin/bash

################################################################################
# Download NBA Data from Kaggle
#
# Simple script for users who want to get started quickly
# Downloads public Kaggle datasets (no authentication required)
#
# Usage:
#   ./scripts/download_kaggle_data.sh
#   ./scripts/download_kaggle_data.sh --dataset username/dataset-name
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default datasets (REPLACE WITH YOUR KAGGLE USERNAME)
KAGGLE_DATASET="YOUR_KAGGLE_USERNAME/nba-game-stats-daily"
PROCESSED_DATASET="YOUR_KAGGLE_USERNAME/nba-processed-data"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            KAGGLE_DATASET="$2"
            shift 2
            ;;
        --processed)
            PROCESSED_DATASET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dataset username/dataset-name] [--processed username/processed-name]"
            echo ""
            echo "Downloads NBA data from Kaggle (public datasets - no auth required)"
            echo ""
            echo "Options:"
            echo "  --dataset    Kaggle dataset ID for cumulative scraped data"
            echo "  --processed  Kaggle dataset ID for processed data"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Downloading NBA data from Kaggle...${NC}"
echo ""

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo -e "${YELLOW}Installing Kaggle CLI...${NC}"
    pip install -q kaggle
fi

# Create directories
mkdir -p data/cumulative_scraped data/processed

# Download cumulative scraped data
echo -e "${BLUE}Downloading cumulative scraped data...${NC}"
echo "Dataset: $KAGGLE_DATASET"
if kaggle datasets download -d "$KAGGLE_DATASET" -p data/cumulative_scraped --unzip; then
    echo -e "${GREEN}✓ Cumulative scraped data downloaded${NC}"

    # Show what was downloaded
    echo "Files:"
    ls -lh data/cumulative_scraped/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (no CSV files found)"
else
    echo -e "${YELLOW}Warning: Could not download cumulative scraped data${NC}"
    echo "Make sure the dataset exists and is public: https://kaggle.com/datasets/$KAGGLE_DATASET"
    exit 1
fi

echo ""

# Download processed data
echo -e "${BLUE}Downloading processed data...${NC}"
echo "Dataset: $PROCESSED_DATASET"
if kaggle datasets download -d "$PROCESSED_DATASET" -p data/processed --unzip; then
    echo -e "${GREEN}✓ Processed data downloaded${NC}"

    # Show what was downloaded
    echo "Files:"
    ls -lh data/processed/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (no CSV files found)"
else
    echo -e "${YELLOW}Warning: Could not download processed data${NC}"
    echo "This is optional - you can generate it by running data processing"
fi

echo ""
echo -e "${GREEN}✓ Data download complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run ML pipeline: ./scripts/run_nightly_pipeline.sh --data-source kaggle"
echo "  2. View dashboard: streamlit run streamlit_app/app.py"
