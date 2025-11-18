#!/usr/bin/env python3
"""
Cross-Platform Pipeline Runner for NBA Prediction System

This script provides a Python-based entry point for running the NBA prediction
pipeline on any platform (Windows, macOS, Linux). It's a thin wrapper around
the Python modules that handles orchestration.

For Linux/macOS users: The bash scripts (run_with_kaggle_data.sh,
run_nightly_pipeline.sh) are preferred for better logging and error handling.

For Windows users: This script works natively without WSL.

Usage:
    # Kaggle workflow
    uv run scripts/run_pipeline.py --source kaggle

    # Full pipeline with webscraping
    uv run scripts/run_pipeline.py --source scraping

    # Skip specific stages
    uv run scripts/run_pipeline.py --source kaggle --skip-dashboard

Environment Variables:
    MLFLOW_TRACKING_URI - MLflow server URI (defaults to local mlruns)
    KAGGLE_USERNAME - Kaggle username (for private datasets)
    KAGGLE_KEY - Kaggle API key (for private datasets)
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Colors for cross-platform output
try:
    import colorama
    colorama.init()
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'
except ImportError:
    GREEN = YELLOW = RED = BLUE = NC = ''

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
LOG_DIR = PROJECT_DIR / "logs"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PIPELINE_LOG = LOG_DIR / f"pipeline_{TIMESTAMP}.log"

# Default Kaggle dataset
KAGGLE_DATASET = "chrismunch/nba-game-team-statistics"


def log_info(message: str) -> None:
    """Log info message."""
    print(f"{BLUE}[INFO]{NC} {message}")


def log_success(message: str) -> None:
    """Log success message."""
    print(f"{GREEN}[SUCCESS]{NC} {message}")


def log_warning(message: str) -> None:
    """Log warning message."""
    print(f"{YELLOW}[WARNING]{NC} {message}")


def log_error(message: str) -> None:
    """Log error message."""
    print(f"{RED}[ERROR]{NC} {message}")


def print_header(stage: str, description: str) -> None:
    """Print stage header."""
    print("\n" + "=" * 80)
    print(f"  STAGE {stage}: {description}")
    print("=" * 80 + "\n")


def run_command(command: List[str], stage_name: str) -> int:
    """
    Run a command and return exit code.

    Args:
        command: Command to run as list of strings
        stage_name: Human-readable stage name for logging

    Returns:
        Exit code (0 = success)
    """
    log_info(f"Running: {' '.join(command)}")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=False,
            capture_output=False  # Let output stream to console
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            log_success(f"{stage_name} completed in {duration:.1f}s")
        else:
            log_error(f"{stage_name} failed after {duration:.1f}s (exit code: {result.returncode})")

        return result.returncode

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_error(f"{stage_name} failed after {duration:.1f}s: {str(e)}")
        return 1


def download_kaggle_data(dataset: str = KAGGLE_DATASET) -> int:
    """Download data from Kaggle."""
    print_header("1", "Download Data from Kaggle")

    # Check if kaggle CLI is available
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log_warning("Kaggle CLI not found, installing...")
        result = subprocess.run(["uv", "pip", "install", "kaggle"], cwd=PROJECT_DIR)
        if result.returncode != 0:
            log_error("Failed to install Kaggle CLI")
            return 1

    # Create directories
    (PROJECT_DIR / "data" / "cumulative_scraped").mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

    log_info(f"Downloading from Kaggle dataset: {dataset}")

    # Download data
    exit_code = run_command(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", "data", "--unzip"],
        "Kaggle Download"
    )

    if exit_code == 0:
        # Show what was downloaded
        log_info("Downloaded files:")
        for pattern in ["cumulative_scraped/*.csv", "processed/*.csv"]:
            for file in PROJECT_DIR.glob(f"data/{pattern}"):
                size_mb = file.stat().st_size / (1024 * 1024)
                log_info(f"  {file.relative_to(PROJECT_DIR)} ({size_mb:.1f} MB)")
    else:
        log_error("Failed to download data from Kaggle")
        log_error(f"Make sure dataset exists: https://kaggle.com/datasets/{dataset}")

    return exit_code


def run_webscraping() -> int:
    """Run webscraping stage."""
    print_header("1", "Webscraping (Schedule & Results)")
    return run_command(
        ["uv", "run", "-m", "src.nba_app.webscraping.main"],
        "Webscraping"
    )


def run_data_processing() -> int:
    """Run data processing stage."""
    print_header("2", "Data Processing (Consolidation & Cleaning)")
    return run_command(
        ["uv", "run", "-m", "src.nba_app.data_processing.main"],
        "Data Processing"
    )


def run_feature_engineering() -> int:
    """Run feature engineering stage."""
    print_header("3", "Feature Engineering (849 Features)")
    return run_command(
        ["uv", "run", "-m", "src.nba_app.feature_engineering.main"],
        "Feature Engineering"
    )


def run_inference() -> int:
    """Run inference stage."""
    print_header("4", "Inference (Predictions with Uncertainty)")
    return run_command(
        ["uv", "run", "-m", "src.nba_app.inference.main"],
        "Inference"
    )


def run_dashboard_prep() -> int:
    """Run dashboard prep stage."""
    print_header("5", "Dashboard Prep (Aggregation & Metrics)")
    return run_command(
        ["uv", "run", "-m", "src.nba_app.dashboard_prep.main"],
        "Dashboard Prep"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-platform NBA prediction pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Kaggle data (recommended)
  uv run scripts/run_pipeline.py --source kaggle

  # Run full pipeline with webscraping
  uv run scripts/run_pipeline.py --source scraping

  # Skip download (use existing data)
  uv run scripts/run_pipeline.py --source kaggle --skip-download

  # Skip dashboard prep
  uv run scripts/run_pipeline.py --source kaggle --skip-dashboard
        """
    )

    parser.add_argument(
        "--source",
        choices=["kaggle", "scraping"],
        default="kaggle",
        help="Data source: kaggle (download from Kaggle) or scraping (scrape NBA.com)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download (use existing local data)"
    )
    parser.add_argument(
        "--skip-webscraping",
        action="store_true",
        help="Skip webscraping stage (use existing scraped data)"
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip dashboard prep stage"
    )
    parser.add_argument(
        "--dataset",
        default=KAGGLE_DATASET,
        help=f"Kaggle dataset ID (default: {KAGGLE_DATASET})"
    )

    args = parser.parse_args()

    # Setup
    LOG_DIR.mkdir(exist_ok=True)

    log_info("NBA Prediction Pipeline Starting")
    log_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"Project Directory: {PROJECT_DIR}")
    log_info(f"Source: {args.source}")

    # Check MLFLOW_TRACKING_URI
    if "MLFLOW_TRACKING_URI" not in os.environ:
        log_warning("MLFLOW_TRACKING_URI not set, using local mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = f"file:///{PROJECT_DIR}/mlruns"
    else:
        log_info(f"MLflow Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")

    start_time = datetime.now()

    try:
        if args.source == "kaggle":
            # Kaggle workflow
            if not args.skip_download:
                exit_code = download_kaggle_data(args.dataset)
                if exit_code != 0:
                    sys.exit(exit_code)
            else:
                log_warning("Skipping Kaggle download - using existing local data")

            # Verify required files
            required_file = PROJECT_DIR / "data" / "processed" / "teams_boxscores.csv"
            if not required_file.exists():
                log_error(f"Required file not found: {required_file}")
                log_error("Please run with download enabled or check data directory")
                sys.exit(1)

            # Run stages (Kaggle workflow doesn't need data processing)
            stages = [
                (run_feature_engineering, 3),
                (run_inference, 4),
            ]

        else:  # scraping
            # Full pipeline workflow
            stages = []

            if not args.skip_webscraping:
                stages.append((run_webscraping, 1))
            else:
                log_warning("Skipping webscraping - using existing scraped data")

            stages.extend([
                (run_data_processing, 2),
                (run_feature_engineering, 3),
                (run_inference, 4),
            ])

        # Add dashboard prep if not skipped
        if not args.skip_dashboard:
            stages.append((run_dashboard_prep, 5))
        else:
            log_warning("Skipping dashboard prep per user request")

        # Run all stages
        for stage_func, stage_num in stages:
            exit_code = stage_func()
            if exit_code != 0:
                log_error(f"Pipeline failed at stage {stage_num}")
                if stage_num == 5:
                    log_warning("Dashboard prep failed, but predictions are still available")
                else:
                    sys.exit(stage_num)

        # Success
        duration = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 80)
        print("  PIPELINE COMPLETE")
        print("=" * 80)
        log_success(f"Pipeline completed successfully in {duration:.1f}s")

        # Show output files
        log_info("\nOutput Files:")

        prediction_files = sorted(
            (PROJECT_DIR / "data" / "predictions").glob("predictions_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if prediction_files:
            latest = prediction_files[0]
            log_success(f"  ✓ {latest.relative_to(PROJECT_DIR)}")

            # Count predictions
            try:
                with open(latest) as f:
                    pred_count = sum(1 for _ in f) - 1  # Subtract header
                log_info(f"    Generated {pred_count} predictions")
            except Exception:
                pass

        if not args.skip_dashboard:
            dashboard_file = PROJECT_DIR / "data" / "dashboard" / "dashboard_data.csv"
            if dashboard_file.exists():
                log_success(f"  ✓ {dashboard_file.relative_to(PROJECT_DIR)}")

        # Next steps
        print()
        log_info("Next Steps:")
        log_info("  • View predictions: cat data/predictions/predictions_*.csv")
        log_info("  • Launch dashboard: uv run streamlit run streamlit_app/app.py")
        log_info("  • Refresh data: Run this script again (daily)")

        log_success("\nAll done! 🏀")

    except KeyboardInterrupt:
        print()
        log_warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
