from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from streamlit_app.di_container import StreamlitAppContainer

# Initialize container and logger
container = StreamlitAppContainer()
_app_logger = container.app_logger()
_app_logger.setup("streamlit_dashboard.log")


@st.cache_data(show_spinner=False)
def load_latest_dataset() -> pd.DataFrame:
    """Fetch the most recent dashboard dataset."""
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


def get_high_confidence_threshold() -> float:
    """Return the configured high win probability threshold."""
    service = container.dashboard_data_service()
    return service.get_high_confidence_threshold()
