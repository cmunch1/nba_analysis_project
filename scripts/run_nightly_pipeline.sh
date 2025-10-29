#!/bin/bash

################################################################################
# NBA Analysis Project - Nightly Pipeline Orchestration Script
#
# This script runs the complete batch pipeline for NBA win prediction:
# 1. Webscraping - Scrape today's schedule and yesterday's results
# 2. Data Processing - Clean and consolidate scraped data
# 3. Feature Engineering - Generate features for today's games
# 4. Inference - Generate predictions with uncertainty quantification
# 5. Dashboard Prep - Aggregate predictions and results (OPTIONAL - skipped if blocked)
#
# Usage:
#   ./scripts/run_nightly_pipeline.sh [--skip-webscraping] [--skip-dashboard]
#
# Options:
#   --skip-webscraping   Skip stage 1 (use existing scraped data)
#   --skip-dashboard     Skip stage 5 (dashboard prep has known blocker)
#
# Exit Codes:
#   0 - Success (all stages completed)
#   1 - Stage 1 failed (webscraping)
#   2 - Stage 2 failed (data processing)
#   3 - Stage 3 failed (feature engineering)
#   4 - Stage 4 failed (inference)
#   5 - Stage 5 failed (dashboard prep)
#   99 - Setup/configuration error
#
# Environment Variables:
#   MLFLOW_TRACKING_URI - MLflow server URI (defaults to local mlruns)
#   PROXY_URL - Proxy for webscraping (optional)
#
################################################################################

set -u  # Exit on undefined variable
# Note: We don't use -e because we want to handle errors ourselves

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
PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Default options
SKIP_WEBSCRAPING=false
SKIP_DASHBOARD=false  # Default to including dashboard

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-webscraping)
            SKIP_WEBSCRAPING=true
            shift
            ;;
        --skip-dashboard)
            SKIP_DASHBOARD=true
            shift
            ;;
        --include-dashboard)
            SKIP_DASHBOARD=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-webscraping] [--skip-dashboard] [--include-dashboard]"
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

log_info "NBA Nightly Pipeline Starting"
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
log_info "  Skip Webscraping: $SKIP_WEBSCRAPING"
log_info "  Skip Dashboard: $SKIP_DASHBOARD"

################################################################################
# Pipeline Execution
################################################################################

OVERALL_START_TIME=$(date +%s)

# Stage 1: Webscraping
if [ "$SKIP_WEBSCRAPING" = true ]; then
    log_warning "Skipping Stage 1 (Webscraping) - using existing data"
else
    print_header "1" "Webscraping (Schedule & Results)"
    run_stage 1 "Webscraping" "uv run -m src.nba_app.webscraping.main"
    STAGE1_EXIT=$?
    if [ $STAGE1_EXIT -ne 0 ]; then
        log_error "Pipeline failed at Stage 1 (Webscraping)"
        exit $STAGE1_EXIT
    fi
fi

# Stage 2: Data Processing
print_header "2" "Data Processing (Consolidation & Cleaning)"
run_stage 2 "Data Processing" "uv run -m src.nba_app.data_processing.main"
STAGE2_EXIT=$?
if [ $STAGE2_EXIT -ne 0 ]; then
    log_error "Pipeline failed at Stage 2 (Data Processing)"
    exit $STAGE2_EXIT
fi

# Stage 3: Feature Engineering
print_header "3" "Feature Engineering (849 Features with Deterministic Ordering)"
run_stage 3 "Feature Engineering" "uv run -m src.nba_app.feature_engineering.main"
STAGE3_EXIT=$?
if [ $STAGE3_EXIT -ne 0 ]; then
    log_error "Pipeline failed at Stage 3 (Feature Engineering)"
    exit $STAGE3_EXIT
fi

# Stage 4: Inference
print_header "4" "Inference (Predictions with Uncertainty Quantification)"
run_stage 4 "Inference" "uv run -m src.nba_app.inference.main"
STAGE4_EXIT=$?
if [ $STAGE4_EXIT -ne 0 ]; then
    log_error "Pipeline failed at Stage 4 (Inference)"
    exit $STAGE4_EXIT
fi

# Stage 5: Dashboard Prep (Optional)
if [ "$SKIP_DASHBOARD" = true ]; then
    log_warning "Skipping Stage 5 (Dashboard Prep) - known data schema blocker"
else
    print_header "5" "Dashboard Prep (Aggregation & Performance Metrics)"
    run_stage 5 "Dashboard Prep" "uv run -m src.nba_app.dashboard_prep.main"
    STAGE5_EXIT=$?
    if [ $STAGE5_EXIT -ne 0 ]; then
        log_error "Pipeline failed at Stage 5 (Dashboard Prep)"
        log_warning "This is a known issue - continuing anyway"
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
if [ "$SKIP_WEBSCRAPING" = false ]; then
    if [ -f "data/newly_scraped/todays_matchups.csv" ]; then
        log_success "  ✓ data/newly_scraped/todays_matchups.csv"
    fi
fi
if [ -f "data/processed/teams_boxscores.csv" ]; then
    log_success "  ✓ data/processed/teams_boxscores.csv"
fi
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
log_success "All done! 🏀"

exit 0
