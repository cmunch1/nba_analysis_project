#!/bin/bash

################################################################################
# Fix MLflow metadata paths for local development
#
# This script fixes hardcoded Docker paths in MLflow metadata files
# when you switch between Docker and local development environments.
#
# Problem: When models are registered in Docker (path: /app/mlruns/...),
#          the storage_location gets hardcoded in meta.yaml files.
#          When running locally, MLflow can't find these paths.
#
# Solution: This script updates all meta.yaml files to use the correct
#          local absolute path.
#
# Usage:
#   ./scripts/fix_mlflow_paths.sh
#
################################################################################

set -e

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[INFO]${NC} Fixing MLflow metadata paths..."
echo -e "${BLUE}[INFO]${NC} Project Directory: $PROJECT_DIR"

# Check if mlruns directory exists
if [ ! -d "$PROJECT_DIR/mlruns" ]; then
    echo -e "${RED}[ERROR]${NC} mlruns directory not found at: $PROJECT_DIR/mlruns"
    exit 1
fi

# Find all meta.yaml files with Docker paths
DOCKER_PATH_FILES=$(find "$PROJECT_DIR/mlruns" -name "meta.yaml" -type f -exec grep -l "file:///app/mlruns" {} \; 2>/dev/null || true)
FILE_COUNT=$(echo "$DOCKER_PATH_FILES" | grep -c "meta.yaml" || echo "0")

if [ "$FILE_COUNT" -eq "0" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} No files with Docker paths found. Nothing to fix!"
    exit 0
fi

echo -e "${YELLOW}[INFO]${NC} Found $FILE_COUNT meta.yaml files with Docker paths"

# Backup the mlruns directory
BACKUP_DIR="$PROJECT_DIR/mlruns_backup_$(date +%Y%m%d_%H%M%S)"
echo -e "${BLUE}[INFO]${NC} Creating backup at: $BACKUP_DIR"
cp -r "$PROJECT_DIR/mlruns" "$BACKUP_DIR"

# Update all meta.yaml files
echo -e "${BLUE}[INFO]${NC} Updating paths..."
find "$PROJECT_DIR/mlruns" -name "meta.yaml" -type f -exec sed -i \
    "s|file:///app/mlruns|file://$PROJECT_DIR/mlruns|g" {} \;

# Verify the fix
REMAINING_DOCKER_PATHS=$(find "$PROJECT_DIR/mlruns" -name "meta.yaml" -type f -exec grep -l "file:///app/mlruns" {} \; 2>/dev/null | wc -l || echo "0")

if [ "$REMAINING_DOCKER_PATHS" -eq "0" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} All paths updated successfully!"
    echo -e "${GREEN}[SUCCESS]${NC} Updated $FILE_COUNT files"
    echo -e "${BLUE}[INFO]${NC} Backup saved at: $BACKUP_DIR"
    echo -e "${YELLOW}[TIP]${NC} You can delete the backup once you verify everything works"
else
    echo -e "${RED}[ERROR]${NC} Some paths were not updated. Check manually."
    exit 1
fi

exit 0
