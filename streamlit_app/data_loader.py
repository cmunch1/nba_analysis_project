from pathlib import Path
from typing import List, Optional
import os

import pandas as pd
import streamlit as st

from streamlit_app.di_container import StreamlitAppContainer

# Initialize container and logger
container = StreamlitAppContainer()
_app_logger = container.app_logger()
_app_logger.setup("streamlit_dashboard.log")


def _get_dashboard_file_mtime() -> float:
    """Get modification time of dashboard data file for cache invalidation."""
    dashboard_path = Path("data/dashboard/dashboard_data.csv")
    if dashboard_path.exists():
        return os.path.getmtime(dashboard_path)
    return 0.0


@st.cache_data(show_spinner=False)
def load_latest_dataset(_file_mtime: float = None) -> pd.DataFrame:
    """
    Fetch the most recent dashboard dataset.

    Args:
        _file_mtime: File modification time (prefixed with _ to hide from cache key display)
    """
    service = container.dashboard_data_service()
    return service.get_latest_dashboard_data()


@st.cache_data(show_spinner=False)
def load_snapshot(path_str: str) -> pd.DataFrame:
    """Fetch a specific archived snapshot."""
    service = container.dashboard_data_service()
    return service.load_snapshot(Path(path_str))


def list_archived_snapshots(limit: Optional[int] = None) -> List[Path]:
    """List archived dashboard snapshots."""
    service = container.dashboard_data_service()
    return service.list_archived_snapshots(limit=limit)


def get_high_probability_threshold() -> float:
    """Return the configured high win probability threshold."""
    service = container.dashboard_data_service()
    return service.get_high_probability_threshold()
