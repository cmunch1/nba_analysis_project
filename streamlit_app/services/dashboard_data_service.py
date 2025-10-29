import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ml_framework.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory


@dataclass
class DashboardDataPaths:
    """Resolved locations for dashboard outputs."""

    latest: Path
    archive_dir: Path


class DashboardDataService:
    """
    Provide dashboard datasets to the Streamlit application using core DI components.
    """

    def __init__(
        self,
        config: BaseConfigManager,
        app_logger: BaseAppLogger,
        app_file_handler: BaseAppFileHandler,
        error_handler: ErrorHandlerFactory,
    ) -> None:
        self.config = config
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler
        self.error_handler = error_handler

    def resolve_paths(self) -> DashboardDataPaths:
        """
        Resolve the latest dashboard CSV path and archive directory.
        """
        output_dir = Path(self.config.dashboard_prep.output.output_dir)
        output_filename = self.config.dashboard_prep.output.output_filename
        archive_dir = Path(self.config.dashboard_prep.output.snapshot_dir)
        latest_path = self.app_file_handler.join_paths(output_dir, output_filename)
        return DashboardDataPaths(latest=Path(latest_path), archive_dir=archive_dir)

    def get_latest_dashboard_data(self) -> pd.DataFrame:
        """
        Load the most recent dashboard dataset produced by the nightly pipeline.
        """
        paths = self.resolve_paths()
        data_path = paths.latest

        self.app_logger.structured_log(
            logging.INFO,
            "Loading latest dashboard dataset",
            data_path=str(data_path),
        )

        try:
            return self.app_file_handler.read_csv(data_path)
        except FileNotFoundError as exc:
            raise self.error_handler.create_error_handler(
                "data_processing",
                "Latest dashboard dataset not found",
                error_message=str(exc),
                traceback=traceback.format_exc(),
                expected_path=str(data_path),
            ) from exc

    def get_high_confidence_threshold(self) -> float:
        """
        Retrieve the configured high win probability threshold for predictions.
        """
        try:
            # Try new config key first
            return float(self.config.dashboard_prep.predictions.high_probability_threshold)
        except AttributeError:
            # Fall back to old config key for backwards compatibility
            try:
                return float(self.config.dashboard_prep.predictions.high_confidence_threshold)
            except AttributeError:
                return 0.35

    def list_archived_snapshots(self, limit: Optional[int] = None) -> List[Path]:
        """
        Return available archived dashboard snapshots ordered by recency.
        """
        paths = self.resolve_paths()
        archive_dir = paths.archive_dir

        if not archive_dir.exists():
            self.app_logger.structured_log(
                logging.WARNING,
                "Dashboard archive directory missing",
                archive_dir=str(archive_dir),
            )
            return []

        snapshots = sorted(
            (file_path for file_path in archive_dir.glob("*.csv") if file_path.is_file()),
            key=lambda path: path.stem,
            reverse=True,
        )

        if limit is not None:
            return snapshots[:limit]
        return snapshots

    def load_snapshot(self, snapshot_path: Path) -> pd.DataFrame:
        """
        Load a specific archived dashboard snapshot.
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Loading dashboard snapshot",
            snapshot_path=str(snapshot_path),
        )

        try:
            return self.app_file_handler.read_csv(snapshot_path)
        except FileNotFoundError as exc:
            raise self.error_handler.create_error_handler(
                "data_processing",
                "Requested dashboard snapshot not found",
                error_message=str(exc),
                traceback=traceback.format_exc(),
                snapshot_path=str(snapshot_path),
            ) from exc
