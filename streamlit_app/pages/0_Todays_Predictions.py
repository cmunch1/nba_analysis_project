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
    get_high_probability_threshold,
    load_latest_dataset,
)


st.set_page_config(
    page_title="Today's Predictions",
    page_icon="🏀",
    layout="wide",
)

# Hide the main "app" page from navigation and add title above nav
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] li:first-child {
        display: none;
    }
    [data-testid="stSidebarNav"]::before {
        content: "NBA Predictions and Analysis";
        display: block;
        font-size: 1.5rem;
        font-weight: 600;
        padding: 1rem 1rem 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def render_header(latest_data: pd.DataFrame, predictions_data: pd.DataFrame) -> None:
    """Render top-of-page summary metrics."""
    st.title("Today's Predictions")

    # Get and display prediction date as subtitle
    available_dates = predictions_data.get("game_date")
    if available_dates is not None and not predictions_data.empty:
        date_series = pd.to_datetime(available_dates, errors='coerce')
        valid_dates = date_series[date_series.notna()].dt.date.unique()
        if len(valid_dates) > 0:
            latest_game_date = max(valid_dates)
            st.subheader(latest_game_date.strftime("%A %B %d, %Y"))
        else:
            st.subheader("Date unavailable")
    else:
        st.subheader("Date unavailable")

    st.markdown("---")

    # Extract metrics from the metrics section
    metrics_data = latest_data[latest_data['section'] == 'metrics'] if 'section' in latest_data.columns else pd.DataFrame()

    # Extract season and 7-day accuracy
    season_accuracy = None
    seven_day_accuracy = None
    total_games_predicted = None

    if not metrics_data.empty:
        season_row = metrics_data[metrics_data['window'] == 'season']
        if not season_row.empty and 'accuracy' in season_row.columns:
            season_accuracy = season_row.iloc[0]['accuracy']

        seven_day_row = metrics_data[metrics_data['window'] == '7day']
        if not seven_day_row.empty and 'accuracy' in seven_day_row.columns:
            seven_day_accuracy = seven_day_row.iloc[0]['accuracy']

        # Get total games from season metrics
        if not season_row.empty and 'games_predicted' in season_row.columns:
            total_games_predicted = season_row.iloc[0]['games_predicted']

    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if season_accuracy:
            st.metric("Season Accuracy", season_accuracy)
        else:
            st.metric("Season Accuracy", "N/A")

    with col2:
        if seven_day_accuracy:
            st.metric("7-Day Accuracy", seven_day_accuracy)
        else:
            st.metric("7-Day Accuracy", "N/A")

    with col3:
        if total_games_predicted:
            st.metric("Total Games Predicted", f"{int(total_games_predicted)}")
        else:
            st.metric("Total Games Predicted", "N/A")

    with col4:
        # Count predictions only
        total_predictions = int(predictions_data.get("game_id", pd.Series()).nunique()) if not predictions_data.empty else 0
        st.metric("Games Today", total_predictions)


def render_snapshot_selector() -> Optional[pd.DataFrame]:
    """
    Historical snapshot selection removed from main page.
    Archive selection moved to Historical Performance page.
    """
    return None


def render_dataset_preview(dataset: pd.DataFrame) -> None:
    """Display a preview table."""
    st.subheader("Dashboard Dataset Preview")

    # Show section breakdown
    if 'section' in dataset.columns:
        section_counts = dataset['section'].value_counts()
        st.caption(f"Total rows: {len(dataset)} | " + " | ".join([f"{section}: {count}" for section, count in section_counts.items()]))

    st.dataframe(dataset.head(20), use_container_width=True)


def main() -> None:
    """Streamlit entry point."""
    try:
        latest_dataset = load_latest_dataset()
    except Exception as exc:  # Streamlit renders exceptions with tracebacks automatically.
        st.error("Unable to load the latest dashboard dataset.")
        st.exception(exc)
        return

    snapshot_override = render_snapshot_selector()
    active_dataset = snapshot_override if snapshot_override is not None else latest_dataset

    # Filter to only show predictions (not results or metrics)
    if 'section' in active_dataset.columns:
        predictions_dataset = active_dataset[active_dataset['section'] == 'predictions'].copy()
    else:
        predictions_dataset = active_dataset

    render_header(latest_dataset, predictions_dataset)

    high_prob_threshold = get_high_probability_threshold()

    filter_state: MatchupFilterState = render_filter_panel(predictions_dataset, high_prob_threshold)
    filtered_dataset = filter_matchups(predictions_dataset, filter_state)

    render_kpi_panel(filtered_dataset, high_prob_threshold)
    render_matchup_grid(filtered_dataset, high_prob_threshold)

    # with st.expander("Preview underlying dashboard data"):
    #     render_dataset_preview(active_dataset)


if __name__ == "__main__":
    main()
