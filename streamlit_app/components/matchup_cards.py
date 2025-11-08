from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from streamlit_app.di_container import StreamlitAppContainer


@dataclass
class MatchupFilterState:
    """User selections applied to matchup filtering."""

    selected_date: Optional[pd.Timestamp]
    selected_teams: List[str]
    min_win_probability: float


def render_filter_panel(
    dataset: pd.DataFrame,
    high_probability_threshold: float,
) -> MatchupFilterState:
    """
    Render simplified sidebar with only probability threshold slider.
    Date and team filters removed as they're unnecessary for today's games.
    """
    with st.sidebar:
        st.subheader("Filter Options")

        min_win_probability = st.slider(
            "Win Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Show only games where predicted winner has at least this probability"
        )

        st.markdown("---")
        st.caption("💡 Navigate to **Historical Performance** for past predictions and analysis")

    return MatchupFilterState(
        selected_date=None,  # No date filtering
        selected_teams=[],    # No team filtering
        min_win_probability=min_win_probability,
    )


def filter_matchups(dataset: pd.DataFrame, filters: MatchupFilterState) -> pd.DataFrame:
    """Apply sidebar filters to the dataset."""
    if dataset.empty:
        return dataset

    filtered = dataset.copy()

    if filters.selected_date is not None and "game_date" in filtered.columns:
        game_dates = pd.to_datetime(filtered["game_date"], errors="coerce").dt.date
        filtered = filtered.loc[game_dates == filters.selected_date.date()]

    if filters.selected_teams:
        team_columns = [col for col in ("home_team", "away_team") if col in filtered.columns]
        if team_columns:
            team_mask = False
            for col in team_columns:
                team_mask = team_mask | filtered[col].isin(filters.selected_teams)
            filtered = filtered.loc[team_mask]

    # Filter by predicted probability threshold
    filtered = filtered.loc[filtered["predicted_probability"].fillna(0.0) >= filters.min_win_probability]

    # Sort by predicted probability (highest first)
    filtered = filtered.sort_values("predicted_probability", ascending=False)

    return filtered.reset_index(drop=True)


def render_matchup_grid(
    dataset: pd.DataFrame,
    high_probability_threshold: float,
) -> None:
    """
    Render matchup cards in a responsive grid.
    """
    if dataset.empty:
        st.info("No matchups match the selected filters.")
        return

    cards_per_row = 2
    total_rows = len(dataset)

    for start_idx in range(0, total_rows, cards_per_row):
        columns = st.columns(cards_per_row)
        for offset, column in enumerate(columns):
            row_idx = start_idx + offset
            if row_idx >= total_rows:
                break

            row = dataset.iloc[row_idx]
            with column:
                _render_card(row, high_probability_threshold)


def _render_card(row: pd.Series, high_probability_threshold: float) -> None:
    """Render a single matchup card."""
    home_team_abbrev = row.get("home_team", "Home")
    away_team_abbrev = row.get("away_team", "Away")
    game_time = row.get("game_time") or row.get("game_date")
    prediction_raw = row.get("predicted_winner", "N/A")
    win_prob = row.get("calibrated_home_win_prob")
    predicted_probability = row.get("predicted_probability")

    # Load team name mapping from config
    container = StreamlitAppContainer()
    config = container.config()

    # Convert SimpleNamespace to dict
    if hasattr(config, 'abbrev_to_full_name'):
        team_names = vars(config.abbrev_to_full_name)
    else:
        team_names = {}

    # Get full team names
    home_team = team_names.get(home_team_abbrev, home_team_abbrev)
    away_team = team_names.get(away_team_abbrev, away_team_abbrev)

    # Convert "home"/"away" to actual team names
    if prediction_raw == "home":
        predicted_team = home_team
    elif prediction_raw == "away":
        predicted_team = away_team
    else:
        predicted_team = prediction_raw

    card = st.container(border=True)

    # Display as "Away @ Home" format - single line with no spacing
    card.markdown(f"**{away_team} @ {home_team}**")

    if game_time:
        card.caption(f"Tip-off: {game_time}")

    # Calculate win probability for the predicted winner
    if pd.notnull(win_prob):
        if prediction_raw == "away":
            winner_prob = 1 - win_prob  # Away team probability
        else:
            winner_prob = win_prob  # Home team probability
    else:
        winner_prob = None

    # Display prediction info with smaller font for values
    info_columns = card.columns(2)

    with info_columns[0]:
        st.caption("Predicted Winner")
        st.markdown(f"<span style='font-size: 1.2em;'>{predicted_team if pd.notnull(predicted_team) else 'N/A'}</span>", unsafe_allow_html=True)

    if winner_prob is not None:
        with info_columns[1]:
            st.caption("Win Probability")
            st.markdown(f"<span style='font-size: 1.2em;'>{winner_prob * 100:0.1f}%</span>", unsafe_allow_html=True)

    chips = _build_reliability_chips(row, high_probability_threshold)
    if chips:
        chip_markdown = " ".join(f"`{chip}`" for chip in chips)
        card.markdown(chip_markdown)

    notes = row.get("notes")
    if pd.notnull(notes):
        card.caption(notes)


def _build_reliability_chips(row: pd.Series, high_probability_threshold: float) -> List[str]:
    chips: List[str] = []

    predicted_probability = row.get("predicted_probability")
    if pd.notnull(predicted_probability) and predicted_probability >= high_probability_threshold:
        chips.append("High Win Probability")

    if pd.notnull(row.get("prediction_error")) and row["prediction_error"] <= 0.2:
        chips.append("Recent Accuracy")

    return chips


def _format_interval(lower: Optional[float], upper: Optional[float]) -> Optional[str]:
    if pd.notnull(lower) and pd.notnull(upper):
        return f"{lower * 100:0.1f}% – {upper * 100:0.1f}%"
    return None
