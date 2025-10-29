from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from streamlit_app.components import (
    MatchupFilterState,
    filter_matchups,
    render_filter_panel,
    render_kpi_panel,
    render_matchup_grid,
)
from streamlit_app.data_loader import (
    get_high_confidence_threshold,
    list_archived_snapshots,
    load_latest_dataset,
    load_snapshot,
)


st.set_page_config(
    page_title="NBA Win Predictions",
    page_icon="🏀",
    layout="wide",
)


def format_snapshot_label(path_str: str) -> str:
    """
    Format snapshot filenames into user-friendly labels.
    """
    snapshot_path = Path(path_str)
    try:
        timestamp_str = snapshot_path.stem.split("_")[-1]
        snapshot_dt = datetime.strptime(timestamp_str, "%Y-%m-%d")
        return snapshot_dt.strftime("%b %d, %Y")
    except (ValueError, IndexError):
        return snapshot_path.name


def render_header(latest_data: pd.DataFrame) -> None:
    """Render top-of-page summary metrics."""
    st.title("NBA Predictions Control Panel")
    st.caption("Streamlit prototype backed by nightly dashboard exports.")

    total_games = int(latest_data.get("game_id", pd.Series()).nunique()) if not latest_data.empty else 0
    available_dates = latest_data.get("game_date")
    if available_dates is not None and not latest_data.empty:
        unique_dates = pd.to_datetime(available_dates).dt.date.unique()
        latest_game_date = max(unique_dates)
        st.metric("Latest Game Date", latest_game_date.strftime("%b %d, %Y"))
    st.metric("Games Covered", total_games)


def render_snapshot_selector() -> Optional[pd.DataFrame]:
    """Allow the user to load a historical snapshot."""
    snapshots = list_archived_snapshots(limit=30)
    if not snapshots:
        return None

    snapshot_options = {format_snapshot_label(str(path)): str(path) for path in snapshots}
    with st.sidebar:
        st.subheader("Archive")
        selected_label = st.selectbox(
            "Historical snapshot",
            options=["Latest dataset"] + list(snapshot_options.keys()),
            index=0,
        )

    if selected_label == "Latest dataset":
        return None

    selected_path = snapshot_options[selected_label]
    return load_snapshot(selected_path)


def render_dataset_preview(dataset: pd.DataFrame) -> None:
    """Display a preview table."""
    st.subheader("Dashboard Dataset Preview")
    st.dataframe(dataset.head(20), use_container_width=True)


def main() -> None:
    """Streamlit entry point."""
    try:
        latest_dataset = load_latest_dataset()
    except Exception as exc:  # Streamlit renders exceptions with tracebacks automatically.
        st.error("Unable to load the latest dashboard dataset.")
        st.exception(exc)
        return

    render_header(latest_dataset)

    snapshot_override = render_snapshot_selector()
    active_dataset = snapshot_override if snapshot_override is not None else latest_dataset
    high_prob_threshold = get_high_confidence_threshold()

    filter_state: MatchupFilterState = render_filter_panel(active_dataset, high_prob_threshold)
    filtered_dataset = filter_matchups(active_dataset, filter_state)

    render_kpi_panel(filtered_dataset, high_prob_threshold)
    render_matchup_grid(filtered_dataset, high_prob_threshold)

    with st.expander("Preview underlying dashboard data"):
        render_dataset_preview(active_dataset)


if __name__ == "__main__":
    main()
