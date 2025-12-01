#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Config ---
PROXY_URL="${PROXY_URL:-}"   # optional; leave empty to disable
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
KAGGLE_DIR="${HOME}/.kaggle"
KAGGLE_FILE="${KAGGLE_DIR}/kaggle.json"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p "$DATA_DIR/cumulative_scraped" "$DATA_DIR/processed" "$LOG_DIR"

if [[ -f "$KAGGLE_FILE" ]]; then
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    # Pull credentials from existing kaggle.json
    read -r KAGGLE_USERNAME KAGGLE_KEY < <(python - <<'PY'
import json, os, sys
path = os.path.expanduser("~/.kaggle/kaggle.json")
try:
    with open(path) as f:
        data = json.load(f)
    print(data.get("username", ""), data.get("key", ""))
except Exception as exc:
    sys.stderr.write(f"Failed to read {path}: {exc}\n")
    sys.exit(1)
PY
) || { echo "Set KAGGLE_USERNAME/KAGGLE_KEY env vars or provide a readable kaggle.json."; exit 1; }
  fi
else
  : "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME env var or mount ~/.kaggle/kaggle.json}"
  : "${KAGGLE_KEY:?Set KAGGLE_KEY env var or mount ~/.kaggle/kaggle.json}"
  mkdir -p "$KAGGLE_DIR"
  echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > "$KAGGLE_FILE"
  chmod 600 "$KAGGLE_FILE"
fi

DATASET_OWNER="${DATASET_OWNER:-$KAGGLE_USERNAME}"

if command -v kaggle >/dev/null 2>&1 && kaggle --version >/dev/null 2>&1; then
  KAGGLE_CMD=( "$(command -v kaggle)" )
else
  KAGGLE_CMD=()
  for PY_EXE in python "$PROJECT_ROOT/.venv/bin/python"; do
    if [[ -x "$PY_EXE" ]] && "$PY_EXE" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
spec = importlib.util.find_spec("kaggle.cli")
sys.exit(0 if spec else 1)
PY
    then
      KAGGLE_CMD=( "$PY_EXE" -m kaggle.cli )
      break
    fi
  done

  if [[ ${#KAGGLE_CMD[@]} -eq 0 ]]; then
    echo "kaggle CLI not installed or not runnable; install with 'pip install kaggle' in the image/venv."
    exit 1
  fi
fi

echo "=== Downloading existing Kaggle dataset (if present) ==="
"${KAGGLE_CMD[@]}" datasets download -d "$DATASET_OWNER"/nba-game-team-statistics -p "$DATA_DIR" --unzip || \
  echo "Download skipped/failed (expected on first run)"

echo "=== Configuring proxy ==="
if [[ -n "$PROXY_URL" ]]; then
  if [[ "$PROXY_URL" =~ ^https?:// ]]; then
    export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL"
    echo "Proxy set to $PROXY_URL"
  else
    echo "Invalid PROXY_URL (must start with http:// or https://). Skipping proxy."
    unset http_proxy https_proxy
  fi
else
  echo "No proxy set"
fi

echo "=== Debug Chromium Installation ==="
which chromium || true
which chromium-browser || true
ls -la /usr/bin/chromium* || true
chromium --version || true
chromedriver --version || true
chromium --headless --no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage --disable-gpu \
  --dump-dom https://www.google.com 2>&1 | head -20 || true
whoami
id

echo "=== Run webscraping ==="
python -m nba_app.webscraping.main

echo "=== Run data processing ==="
python -m nba_app.data_processing.main

echo "=== Preparing for Kaggle upload ==="
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

echo "=== Uploading to Kaggle ==="
"${KAGGLE_CMD[@]}" datasets version -p . -m "Nightly update: $(date +%Y-%m-%d)" -r zip -d

echo "✓ Dataset uploaded successfully!"
