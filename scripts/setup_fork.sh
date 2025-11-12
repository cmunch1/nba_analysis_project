#!/bin/bash

################################################################################
# Setup Script for Forked Repository
#
# This script helps people who fork the project get started quickly
# It guides them through choosing a data source and installing dependencies
#
# Usage:
#   ./scripts/setup_fork.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}============================================${NC}"
echo -e "${BOLD}${BLUE}  NBA Prediction Project - Setup Wizard${NC}"
echo -e "${BOLD}${BLUE}============================================${NC}"
echo ""

# Check if this is a fork
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [[ $REMOTE_URL == *"YOUR_GITHUB_USERNAME/nba_analysis_project"* ]]; then
    echo -e "${YELLOW}⚠️  Note: This appears to be the original repository${NC}"
    echo -e "${YELLOW}   If you're a contributor, you may want to fork it first${NC}"
else
    echo -e "${GREEN}✓ Fork detected - great!${NC}"
fi
echo ""

# Step 1: Install dependencies
echo -e "${BOLD}Step 1: Installing Dependencies${NC}"
echo "This project uses 'uv' for fast Python package management"
echo ""

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
else
    echo -e "${GREEN}✓ uv already installed${NC}"
fi

echo "Installing project dependencies..."
uv sync
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Choose data source
echo -e "${BOLD}Step 2: Choose Data Source${NC}"
echo ""
echo "You have three options:"
echo "  1) ${GREEN}Kaggle${NC} - Download public datasets (recommended for getting started)"
echo "  2) ${YELLOW}Local${NC} - Use data already in the repository (if committed)"
echo "  3) ${RED}Scrape${NC} - Scrape fresh data yourself (requires proxy)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Setting up with Kaggle data...${NC}"

        # Check if kaggle is installed
        if ! command -v kaggle &> /dev/null; then
            echo "Installing Kaggle CLI..."
            pip install kaggle
        fi

        # Download data
        if [ -x "scripts/download_kaggle_data.sh" ]; then
            ./scripts/download_kaggle_data.sh
        else
            echo -e "${YELLOW}Warning: download_kaggle_data.sh not found or not executable${NC}"
            echo "Downloading manually..."
            mkdir -p data/cumulative_scraped data/processed
            echo "Run: kaggle datasets download -d YOUR_KAGGLE_USERNAME/nba-game-stats-daily -p data/cumulative_scraped --unzip"
        fi

        DATA_SOURCE="kaggle"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}Using local data from repository${NC}"

        # Check if data exists
        if [ -d "data/cumulative_scraped" ] && [ -n "$(ls -A data/cumulative_scraped 2>/dev/null)" ]; then
            echo -e "${GREEN}✓ Found data in data/cumulative_scraped/${NC}"
        else
            echo -e "${RED}⚠️  Warning: No data found in data/cumulative_scraped/${NC}"
            echo "You may need to:"
            echo "  - Download from Kaggle (option 1)"
            echo "  - Scrape fresh data (option 3)"
            echo "  - Or commit data to your fork"
        fi

        DATA_SOURCE="local"
        ;;
    3)
        echo ""
        echo -e "${RED}Setting up for web scraping${NC}"
        echo ""
        echo "⚠️  Important: NBA.com blocks direct scraping. You need:"
        echo "  1. A proxy service (e.g., BrightData, Oxylabs, ScraperAPI)"
        echo "  2. Proxy credentials in PROXY_URL environment variable"
        echo ""

        read -p "Do you have a proxy set up? [y/N]: " has_proxy

        if [[ $has_proxy =~ ^[Yy]$ ]]; then
            echo ""
            echo "Great! Make sure to set your PROXY_URL:"
            echo "  export PROXY_URL='http://username:password@proxy-host:port'"
            echo ""
            echo "Then run:"
            echo "  ./scripts/run_nightly_pipeline.sh --data-source scrape"

            DATA_SOURCE="scrape"
        else
            echo ""
            echo -e "${YELLOW}No proxy available. Falling back to Kaggle data for now...${NC}"
            echo ""

            if [ -x "scripts/download_kaggle_data.sh" ]; then
                ./scripts/download_kaggle_data.sh
            fi

            DATA_SOURCE="kaggle"
        fi
        ;;
    *)
        echo "Invalid choice. Defaulting to Kaggle data."
        DATA_SOURCE="kaggle"
        ;;
esac

echo ""

# Step 3: Test the setup
echo -e "${BOLD}Step 3: Testing Setup${NC}"
echo ""
echo "Would you like to run a quick test of the ML pipeline?"
read -p "Run test? [y/N]: " run_test

if [[ $run_test =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running ML pipeline test (this may take a few minutes)..."

    if [ "$DATA_SOURCE" = "scrape" ]; then
        echo "Using scraping mode - make sure PROXY_URL is set!"
        ./scripts/run_nightly_pipeline.sh --data-source scrape
    else
        ./scripts/run_nightly_pipeline.sh --data-source "$DATA_SOURCE"
    fi

    echo ""
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test completed successfully!${NC}"
    else
        echo -e "${RED}✗ Test failed. Check logs/ directory for details.${NC}"
    fi
fi

echo ""
echo -e "${BOLD}${GREEN}============================================${NC}"
echo -e "${BOLD}${GREEN}  Setup Complete!${NC}"
echo -e "${BOLD}${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  ${BOLD}Run ML Pipeline:${NC}"
echo "    ./scripts/run_nightly_pipeline.sh --data-source $DATA_SOURCE"
echo ""
echo "  ${BOLD}View Dashboard:${NC}"
echo "    streamlit run streamlit_app/app.py"
echo ""
echo "  ${BOLD}Run with Docker:${NC}"
echo "    docker-compose up nba-pipeline"
echo ""
echo "  ${BOLD}Test GitHub Actions:${NC}"
echo "    Go to Actions → 'Local Development' → Run workflow"
echo ""
echo "Documentation:"
echo "  - README.md - Project overview"
echo "  - docs/DOCKER.md - Docker deployment guide"
echo "  - DEPLOYMENT_PLAN.md - Full deployment strategy"
echo ""
echo -e "${GREEN}Happy coding! 🏀${NC}"
