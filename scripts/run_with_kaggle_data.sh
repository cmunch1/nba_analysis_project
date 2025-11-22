#!/bin/bash

################################################################################
# NBA Analysis Project - Kaggle Data Pipeline
#
# This script runs the inference pipeline using pre-scraped data from Kaggle:
# 1. Download data from Kaggle (cumulative_scraped + processed)
# 2. Feature Engineering - Generate features for today's games
# 3. Inference - Generate predictions with uncertainty quantification
# 4. Dashboard Prep - Aggregate predictions and results
#
# Use this script if you want to:
# - Avoid webscraping (use maintained Kaggle dataset)
# - Run predictions on a forked repo without secrets
# - Focus on ML inference without data collection overhead
#
# Usage:
#   ./scripts/run_with_kaggle_data.sh [--skip-download] [--skip-dashboard]
#
# Options:
#   --skip-download      Skip Kaggle download (use existing local data)
#   --skip-dashboard     Skip dashboard prep stage
#   --dataset            Kaggle dataset ID (default: chrismunch/nba-game-team-statistics)
#
# Exit Codes:
#   0 - Success (all stages completed)
#   1 - Download failed (Kaggle data unavailable)
#   3 - Stage 3 failed (feature engineering)
#   4 - Stage 4 failed (inference)
#   5 - Stage 5 failed (dashboard prep)
#   99 - Setup/configuration error
#
# Environment Variables:
#   MLFLOW_TRACKING_URI - MLflow server URI (defaults to local mlruns)
#   KAGGLE_USERNAME - Kaggle username (for private datasets)
#   KAGGLE_KEY - Kaggle API key (for private datasets)
#
################################################################################

set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="${LOG_DIR}/kaggle_pipeline_${TIMESTAMP}.log"

# Default options
SKIP_DOWNLOAD=false
SKIP_DASHBOARD=false
KAGGLE_DATASET="chrismunch/nba-game-team-statistics"
DOCKER_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-dashboard)
            SKIP_DASHBOARD=true
            shift
            ;;
        --dataset)
            KAGGLE_DATASET="$2"
            shift 2
            ;;
        --docker)
            DOCKER_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-download] [--skip-dashboard] [--dataset dataset-id]"
            echo ""
            echo "Run NBA prediction pipeline using Kaggle data (no webscraping)"
            echo ""
            echo "Options:"
            echo "  --skip-download   Skip Kaggle download (use existing local data)"
            echo "  --skip-dashboard  Skip dashboard prep stage"
            echo "  --dataset         Kaggle dataset ID (default: chrismunch/nba-game-team-statistics)"
            echo "  --docker          Run in Docker mode (use python directly instead of uv run)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Full pipeline with Kaggle download"
            echo "  $0 --skip-download                    # Use existing local data"
            echo "  $0 --dataset your-username/your-data  # Use custom Kaggle dataset"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 99
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$PIPELINE_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$PIPELINE_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$PIPELINE_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$PIPELINE_LOG"
}

print_header() {
    local stage=$1
    local description=$2
    echo "" | tee -a "$PIPELINE_LOG"
    echo "================================================================================" | tee -a "$PIPELINE_LOG"
    echo "  STAGE $stage: $description" | tee -a "$PIPELINE_LOG"
    echo "================================================================================" | tee -a "$PIPELINE_LOG"
}

run_stage() {
    local stage_num=$1
    local stage_name=$2
    local command=$3

    local start_time=$(date +%s)

    log_info "Running: $command"

    # Run command and capture output
    if eval "$command" >> "$PIPELINE_LOG" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Stage $stage_num ($stage_name) completed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Stage $stage_num ($stage_name) failed after ${duration}s"
        log_error "Check log file: $PIPELINE_LOG"
        return $stage_num
    fi
}

################################################################################
# Pre-flight Checks
################################################################################

log_info "NBA Kaggle Pipeline Starting"
log_info "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Project Directory: $PROJECT_DIR"
log_info "Log File: $PIPELINE_LOG"

# Check we're in the right directory
cd "$PROJECT_DIR" || {
    log_error "Failed to change to project directory: $PROJECT_DIR"
    exit 99
}

# Check uv is available
if ! command -v uv &> /dev/null; then
    log_error "uv command not found. Please install uv first."
    exit 99
fi

# Check MLFLOW_TRACKING_URI
if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
    log_warning "MLFLOW_TRACKING_URI not set, using local mlruns"
    export MLFLOW_TRACKING_URI="file:///${PROJECT_DIR}/mlruns"
else
    log_info "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
fi

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Display configuration
log_info "Configuration:"
log_info "  Kaggle Dataset: $KAGGLE_DATASET"
log_info "  Skip Download: $SKIP_DOWNLOAD"
log_info "  Skip Dashboard: $SKIP_DASHBOARD"

################################################################################
# Pipeline Execution
################################################################################

OVERALL_START_TIME=$(date +%s)

# Stage 1: Download from Kaggle
if [ "$SKIP_DOWNLOAD" = true ]; then
    log_warning "Skipping Kaggle download - using existing local data"
else
    print_header "1" "Download Data from Kaggle"

    # Clear existing data directories to avoid LFS pointer conflicts
    log_info "Clearing existing data directories..."
    rm -rf data/cumulative_scraped data/processed
    mkdir -p data/cumulative_scraped data/processed

    log_info "Downloading from Kaggle dataset: $KAGGLE_DATASET"

    # Check if Kaggle credentials are available
    if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
        log_info "Using Kaggle API with credentials"

        # Find kaggle CLI - prefer system install over virtualenv
        KAGGLE_CMD=""
        if [ -x "/usr/local/bin/kaggle" ]; then
            KAGGLE_CMD="/usr/local/bin/kaggle"
        elif command -v kaggle &> /dev/null; then
            KAGGLE_CMD="kaggle"
        else
            log_warning "Kaggle CLI not found, installing..."
            pip install kaggle >> "$PIPELINE_LOG" 2>&1
            KAGGLE_CMD="kaggle"
        fi

        # Download using Kaggle CLI
        set -o pipefail
        if $KAGGLE_CMD datasets download -d "$KAGGLE_DATASET" -p data --unzip 2>&1 | tee -a "$PIPELINE_LOG"; then
            log_success "Stage 1 (Kaggle Download via API) completed"
        else
            log_error "Failed to download data from Kaggle API"
            log_error "See error above for details"
            set +o pipefail
            exit 1
        fi
        set +o pipefail
    else
        log_info "No Kaggle credentials found, using direct HTTP download"

        # Download using curl (no authentication required for public datasets)
        DOWNLOAD_URL="https://www.kaggle.com/api/v1/datasets/download/${KAGGLE_DATASET}"
        TEMP_ZIP="/tmp/kaggle_data.zip"

        log_info "Downloading from: $DOWNLOAD_URL"

        if curl -L -o "$TEMP_ZIP" "$DOWNLOAD_URL" >> "$PIPELINE_LOG" 2>&1; then
            # Unzip to data directory
            if unzip -o "$TEMP_ZIP" -d data >> "$PIPELINE_LOG" 2>&1; then
                rm -f "$TEMP_ZIP"
                log_success "Stage 1 (Kaggle Download via HTTP) completed"
            else
                log_error "Failed to unzip downloaded data"
                rm -f "$TEMP_ZIP"
                exit 1
            fi
        else
            log_error "Failed to download data from Kaggle"
            log_error "Make sure dataset exists and is public: https://kaggle.com/datasets/$KAGGLE_DATASET"
            exit 1
        fi
    fi

    # Show what was downloaded
    log_info "Downloaded files:"
    if [ -d "data/cumulative_scraped" ]; then
        ls -lh data/cumulative_scraped/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    fi
    if [ -d "data/processed" ]; then
        ls -lh data/processed/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    fi
fi

# Verify required files exist
log_info "Verifying required data files..."
REQUIRED_FILES=(
    "data/processed/teams_boxscores.csv"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Required file not found: $file"
        log_error "Please run with download enabled or check data directory"
        exit 1
    fi
done
log_success "All required data files present"

# Set Python command based on mode
if [ "$DOCKER_MODE" = true ]; then
    PYTHON_CMD="python -m"
    # Set PYTHONPATH to include src directory for module imports
    export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
    log_info "Running in Docker mode (using python directly)"
    log_info "PYTHONPATH: $PYTHONPATH"
else
    PYTHON_CMD="uv run -m src."
fi

# Stage 2: Feature Engineering
print_header "2" "Feature Engineering (849 Features with Deterministic Ordering)"
run_stage 3 "Feature Engineering" "$PYTHON_CMD nba_app.feature_engineering.main"
STAGE3_EXIT=$?
if [ $STAGE3_EXIT -ne 0 ]; then
    log_error "Pipeline failed at Stage 2 (Feature Engineering)"
    exit $STAGE3_EXIT
fi

# Stage 3: Inference
print_header "3" "Inference (Predictions with Uncertainty Quantification)"
run_stage 4 "Inference" "$PYTHON_CMD nba_app.inference.main"
STAGE4_EXIT=$?
if [ $STAGE4_EXIT -ne 0 ]; then
    log_error "Pipeline failed at Stage 3 (Inference)"
    exit $STAGE4_EXIT
fi

# Stage 4: Dashboard Prep (Optional)
if [ "$SKIP_DASHBOARD" = true ]; then
    log_warning "Skipping Stage 4 (Dashboard Prep) per user request"
else
    print_header "4" "Dashboard Prep (Aggregation & Performance Metrics)"
    run_stage 5 "Dashboard Prep" "$PYTHON_CMD nba_app.dashboard_prep.main"
    STAGE5_EXIT=$?
    if [ $STAGE5_EXIT -ne 0 ]; then
        log_error "Pipeline failed at Stage 4 (Dashboard Prep)"
        log_warning "Continuing anyway - predictions are still available"
        # Don't exit, just warn
    fi
fi

################################################################################
# Pipeline Summary
################################################################################

OVERALL_END_TIME=$(date +%s)
OVERALL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))

echo "" | tee -a "$PIPELINE_LOG"
echo "================================================================================" | tee -a "$PIPELINE_LOG"
echo "  PIPELINE COMPLETE" | tee -a "$PIPELINE_LOG"
echo "================================================================================" | tee -a "$PIPELINE_LOG"
log_success "Pipeline completed successfully in ${OVERALL_DURATION}s"
log_info "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

# List output files
log_info "Output Files:"
if [ -f "data/engineered/engineered_features.csv" ]; then
    log_success "  ✓ data/engineered/engineered_features.csv"
fi

# Find latest predictions file
LATEST_PREDICTION=$(ls -t data/predictions/predictions_*.csv 2>/dev/null | head -1)
if [ -n "$LATEST_PREDICTION" ]; then
    log_success "  ✓ $LATEST_PREDICTION"
    # Show prediction count
    PRED_COUNT=$(wc -l < "$LATEST_PREDICTION")
    log_info "    Generated $((PRED_COUNT - 1)) predictions"
fi

if [ "$SKIP_DASHBOARD" = false ] && [ -f "data/dashboard/dashboard_data.csv" ]; then
    log_success "  ✓ data/dashboard/dashboard_data.csv"
fi

log_info "Full log available at: $PIPELINE_LOG"

# Next steps
echo "" | tee -a "$PIPELINE_LOG"
log_info "Next Steps:"
log_info "  • View predictions: cat $LATEST_PREDICTION"
log_info "  • Launch dashboard: uv run streamlit run streamlit_app/app.py"
log_info "  • Refresh data: Run this script again (daily)"

log_success "All done! 🏀"

exit 0
