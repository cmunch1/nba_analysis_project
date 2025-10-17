"""
Script to extract optimal feature lists from pruning analysis and save as YAML allowlist.

This script:
1. Loads the most recent pruning comparison report from the configured directory
2. Checks if pruning was accepted
3. Loads the corresponding feature audit
4. Extracts the optimal feature list (features that were kept)
5. Saves as YAML allowlist for use in feature engineering

Usage:
    python -m src.ml_framework.model_testing.feature_pruning.save_pruned_features [--comparison-file PATH]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from ml_framework.core.common_di_container import CommonDIContainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract optimal feature list from pruning analysis"
    )
    parser.add_argument(
        "--comparison-file",
        type=str,
        help="Path to specific pruning comparison CSV file. If not provided, uses the most recent file from config directory."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save the feature allowlist YAML. If not provided, saves to ml_artifacts/features/allowlists/feature_allowlist_latest.yaml"
    )
    return parser.parse_args()


def get_latest_comparison_file(reports_dir: Path, app_logger) -> Optional[Path]:
    """Get the most recent pruning comparison file from the reports directory."""
    if not reports_dir.exists():
        app_logger.structured_log(
            logging.ERROR,
            "Pruning reports directory does not exist",
            reports_dir=str(reports_dir)
        )
        return None

    # Find all comparison CSV files
    comparison_files = list(reports_dir.glob("*_pruning_comparison_*.csv"))

    if not comparison_files:
        app_logger.structured_log(
            logging.ERROR,
            "No pruning comparison files found in directory",
            reports_dir=str(reports_dir)
        )
        return None

    # Sort by modification time, most recent first
    latest_file = max(comparison_files, key=lambda p: p.stat().st_mtime)

    app_logger.structured_log(
        logging.INFO,
        "Found latest pruning comparison file",
        file_path=str(latest_file),
        modified_time=datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
    )

    return latest_file


def extract_features_from_audit(audit_file: Path, app_logger, app_file_handler) -> Optional[List[str]]:
    """Extract feature names from the audit CSV."""
    try:
        # Read audit CSV using app_file_handler
        df = app_file_handler.read_csv(audit_file)

        if 'feature_name' not in df.columns:
            app_logger.structured_log(
                logging.ERROR,
                "Audit file missing 'feature_name' column",
                audit_file=str(audit_file),
                columns=list(df.columns)
            )
            return None

        # Get all feature names
        features = df['feature_name'].tolist()

        app_logger.structured_log(
            logging.INFO,
            "Extracted features from audit file",
            audit_file=str(audit_file),
            num_features=len(features)
        )

        return features

    except Exception as e:
        app_logger.structured_log(
            logging.ERROR,
            "Failed to extract features from audit file",
            audit_file=str(audit_file),
            error=str(e)
        )
        return None


def get_pruned_features_from_comparison(comparison_file: Path, app_logger, app_file_handler) -> Optional[List[str]]:
    """Extract the list of features that were dropped from the comparison file."""
    try:
        # Read comparison CSV using app_file_handler
        df = app_file_handler.read_csv(comparison_file)

        # Get the pruned features list (should be in the summary section)
        # Look for the row with 'features_dropped'
        if 'metric' in df.columns and 'value' in df.columns:
            dropped_features_row = df[df['metric'] == 'features_dropped']
            if not dropped_features_row.empty:
                # The value might be a list-like string
                dropped_features_str = dropped_features_row.iloc[0]['value']
                # Parse it - it might be something like "[feat1, feat2, feat3]"
                if isinstance(dropped_features_str, str):
                    # Remove brackets and split by comma
                    dropped_features_str = dropped_features_str.strip('[]')
                    dropped_features = [f.strip().strip("'\"") for f in dropped_features_str.split(',')]

                    app_logger.structured_log(
                        logging.INFO,
                        "Extracted dropped features from comparison file",
                        comparison_file=str(comparison_file),
                        num_dropped=len(dropped_features)
                    )

                    return dropped_features

        app_logger.structured_log(
            logging.WARNING,
            "Could not find features_dropped in comparison file",
            comparison_file=str(comparison_file)
        )
        return None

    except Exception as e:
        app_logger.structured_log(
            logging.ERROR,
            "Failed to extract dropped features from comparison file",
            comparison_file=str(comparison_file),
            error=str(e)
        )
        return None


def main():
    """Main execution function."""
    args = parse_args()

    # Initialize DI container
    container = CommonDIContainer()
    container.wire(modules=[sys.modules[__name__]])

    config = container.config()
    app_logger = container.app_logger()
    app_file_handler = container.app_file_handler()

    # Setup logger
    app_logger.setup(config.core.app_logging_config.model_testing_log_file)

    app_logger.structured_log(logging.INFO, "Starting pruned features extraction")

    try:
        # Get directories - config already resolves ${PROJECT_ROOT} but may resolve it incorrectly
        # So we manually use CWD as the actual project root
        project_root = Path.cwd()

        # Use the directory names from config but with correct project root
        reports_dir = project_root / "ml_artifacts" / "features" / "pruning" / "reports"
        audits_dir = project_root / "ml_artifacts" / "features" / "audits"

        app_logger.structured_log(
            logging.INFO,
            "Using configured directories",
            reports_dir=str(reports_dir),
            audits_dir=str(audits_dir)
        )

        # Get comparison file
        if args.comparison_file:
            comparison_file = Path(args.comparison_file)
            if not comparison_file.exists():
                app_logger.structured_log(
                    logging.ERROR,
                    "Specified comparison file does not exist",
                    file_path=str(comparison_file)
                )
                sys.exit(1)
        else:
            comparison_file = get_latest_comparison_file(reports_dir, app_logger)
            if comparison_file is None:
                sys.exit(1)

        # Load comparison report to check recommendation
        df_comparison = app_file_handler.read_csv(comparison_file)

        # Find recommendation - it's in the 'baseline' column for the recommendation row
        recommendation_row = df_comparison[df_comparison['metric'] == 'recommendation']
        if recommendation_row.empty:
            app_logger.structured_log(
                logging.ERROR,
                "No recommendation found in comparison file",
                comparison_file=str(comparison_file)
            )
            sys.exit(1)

        recommendation = recommendation_row.iloc[0]['baseline']

        app_logger.structured_log(
            logging.INFO,
            "Pruning recommendation",
            recommendation=recommendation,
            comparison_file=str(comparison_file)
        )

        if recommendation != 'ACCEPT_PRUNED':
            app_logger.structured_log(
                logging.WARNING,
                "Pruning was not accepted. Feature allowlist will not be created.",
                recommendation=recommendation
            )
            print(f"\nPruning recommendation: {recommendation}")
            print("Feature allowlist will NOT be created (only created for ACCEPT_PRUNED).")
            sys.exit(0)

        # Extract timestamp from comparison filename to find matching audit file
        # Format: xgboost_validation_pruning_comparison_YYYYMMDD_HHMMSS.csv
        filename = comparison_file.stem
        parts = filename.split('_')

        # The last two parts are date and time
        try:
            if len(parts) < 2:
                raise ValueError(f"Unexpected filename format: {filename}")

            date_part = parts[-2]  # YYYYMMDD
            time_part = parts[-1]  # HHMMSS

            # Extract model type (xgboost or lightgbm) and run type (validation or oof)
            # Pattern: <model>_<run_type>_pruning_comparison_<date>_<time>
            model_type = parts[0]  # xgboost or lightgbm
            run_type = parts[1]    # validation or oof

            app_logger.structured_log(
                logging.INFO,
                "Extracted timestamp from comparison filename",
                date=date_part,
                time=time_part,
                model_type=model_type,
                run_type=run_type
            )

            # Find matching audit file - looking for the baseline (earlier timestamp)
            # Format: feature_audit_<model>_<run_type>_<date>_<time>.csv
            audit_pattern = f"feature_audit_{model_type}_{run_type}_{date_part}_*.csv"
            audit_files = list(audits_dir.glob(audit_pattern))

            if not audit_files:
                app_logger.structured_log(
                    logging.ERROR,
                    "No matching audit files found",
                    audit_pattern=audit_pattern,
                    audits_dir=str(audits_dir)
                )
                sys.exit(1)

            # Sort by timestamp (filename) to get baseline (earlier) audit
            # We want the baseline (first run with all features), not the pruned run
            audit_files_sorted = sorted(audit_files, key=lambda p: p.stem)

            # The baseline audit should have an earlier timestamp than the comparison
            # Find the audit file that's closest to but earlier than the comparison timestamp
            comparison_timestamp = f"{date_part}_{time_part}"
            baseline_audit = None
            for audit_f in audit_files_sorted:
                # Extract timestamp from audit filename
                audit_parts = audit_f.stem.split('_')
                audit_timestamp = f"{audit_parts[-2]}_{audit_parts[-1]}"
                if audit_timestamp < comparison_timestamp:
                    baseline_audit = audit_f

            if baseline_audit is None:
                # Fallback to first audit file
                baseline_audit = audit_files_sorted[0]

            audit_file = baseline_audit

            app_logger.structured_log(
                logging.INFO,
                "Found baseline audit file",
                audit_file=str(audit_file),
                num_matching_audits=len(audit_files)
            )

        except (ValueError, IndexError) as e:
            app_logger.structured_log(
                logging.ERROR,
                "Failed to extract timestamp from comparison filename",
                filename=filename,
                error=str(e)
            )
            sys.exit(1)

        # Load the audit file to get features and drop scores
        df_audit = app_file_handler.read_csv(audit_file)

        if 'feature_name' not in df_audit.columns or 'drop_candidate_score' not in df_audit.columns:
            app_logger.structured_log(
                logging.ERROR,
                "Audit file missing required columns",
                audit_file=str(audit_file),
                columns=list(df_audit.columns)
            )
            sys.exit(1)

        # Get drop threshold from config
        model_testing_cfg = config.core.model_testing_config
        drop_threshold = model_testing_cfg.feature_pruning.drop_candidate_threshold

        # Split features into kept (drop_score < threshold) and dropped (drop_score >= threshold)
        kept_features_df = df_audit[df_audit['drop_candidate_score'] < drop_threshold]
        dropped_features_df = df_audit[df_audit['drop_candidate_score'] >= drop_threshold]

        all_features = df_audit['feature_name'].tolist()
        kept_features = kept_features_df['feature_name'].tolist()
        dropped_features = dropped_features_df['feature_name'].tolist()

        app_logger.structured_log(
            logging.INFO,
            "Extracted features from audit file",
            audit_file=str(audit_file),
            num_features=len(all_features),
            drop_threshold=drop_threshold
        )

        app_logger.structured_log(
            logging.INFO,
            "Calculated feature allowlist",
            total_features=len(all_features),
            features_dropped=len(dropped_features),
            features_kept=len(kept_features)
        )

        # Determine output file
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            # Default to ml_artifacts/features/allowlists/feature_allowlist_latest.yaml
            output_file = project_root / "ml_artifacts" / "features" / "allowlists" / "feature_allowlist_latest.yaml"

        # Create output directory if needed
        app_file_handler.ensure_directory(output_file.parent)

        # Create YAML structure
        allowlist_data = {
            'feature_allowlist': {
                'features': sorted(kept_features),  # Sort for consistency
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'source_audit': str(audit_file.name),
                    'source_comparison': str(comparison_file.name),
                    'total_features': len(all_features),
                    'features_dropped': len(dropped_features),
                    'features_kept': len(kept_features),
                    'drop_threshold': drop_threshold,
                    'recommendation': recommendation
                }
            }
        }

        # Save using app_file_handler
        app_file_handler.write_yaml(allowlist_data, output_file)

        app_logger.structured_log(
            logging.INFO,
            "Successfully saved feature allowlist",
            output_file=str(output_file),
            num_features=len(kept_features)
        )

        print(f"\n✓ Feature allowlist created successfully!")
        print(f"  Output file: {output_file}")
        print(f"  Features kept: {len(kept_features)}")
        print(f"  Features dropped: {len(dropped_features)}")
        print(f"\nTo enable the allowlist, edit configs/nba/feature_engineering_config.yaml")
        print(f"and set feature_allowlist.enabled: true")

    except Exception as e:
        import traceback
        app_logger.structured_log(
            logging.ERROR,
            "Failed to extract feature allowlist",
            error=str(e),
            traceback=traceback.format_exc()
        )
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
