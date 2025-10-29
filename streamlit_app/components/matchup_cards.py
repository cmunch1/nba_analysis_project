from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import streamlit as st


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
    Render sidebar controls for matchup filtering and return the current state.
    """
    available_dates = _extract_unique_dates(dataset)
    available_teams = _extract_unique_teams(dataset)

    with st.sidebar:
        st.subheader("Filter Matchups")

        selected_date = None
        if available_dates:
            default_date = max(available_dates)
            label_to_date = {date.strftime("%b %d, %Y"): date for date in available_dates}
            date_label = st.selectbox(
                "Game date",
                options=["All dates"] + list(label_to_date.keys()),
                index=(list(label_to_date.keys()).index(default_date.strftime("%b %d, %Y")) + 1)
                if default_date is not None
                else 0,
            )
            if date_label != "All dates":
                selected_date = label_to_date[date_label]

        selected_teams: List[str] = []
        if available_teams:
            selected_teams = st.multiselect(
                "Teams",
                options=sorted(available_teams),
                default=[],
            )

        default_threshold = float(min(max(high_probability_threshold, 0.0), 1.0))
        min_win_probability = st.slider(
            "Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )

    return MatchupFilterState(
        selected_date=selected_date,
        selected_teams=selected_teams,
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

    if "confidence" in filtered.columns:
        filtered = filtered.loc[filtered["confidence"].fillna(0.0) >= filters.min_win_probability]

    # Display predictions only for rows that include a probability column.
    if "calibrated_home_win_prob" in filtered.columns:
        filtered = filtered.sort_values("calibrated_home_win_prob", ascending=False)

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
    home_team = row.get("home_team", "Home")
    away_team = row.get("away_team", "Away")
    game_time = row.get("game_time") or row.get("game_date")
    prediction_raw = row.get("predicted_winner", "N/A")
    win_prob = row.get("calibrated_home_win_prob")
    confidence = row.get("confidence")

    # Convert "home"/"away" to actual team names
    if prediction_raw == "home":
        predicted_team = home_team
    elif prediction_raw == "away":
        predicted_team = away_team
    else:
        predicted_team = prediction_raw

    card = st.container(border=True)
    card.subheader(f"{home_team} vs {away_team}")
    if game_time:
        card.caption(f"Tip-off: {game_time}")

    info_columns = card.columns(2)
    info_columns[0].metric("Predicted Winner", predicted_team if pd.notnull(predicted_team) else "N/A")

    if pd.notnull(win_prob):
        info_columns[1].metric(
            "Home Win Probability", f"{win_prob * 100:0.1f}%"
        )

    chips = _build_reliability_chips(row, high_probability_threshold)
    if chips:
        chip_markdown = " ".join(f"`{chip}`" for chip in chips)
        card.markdown(chip_markdown)

    notes = row.get("notes")
    if pd.notnull(notes):
        card.caption(notes)


def _build_reliability_chips(row: pd.Series, high_probability_threshold: float) -> List[str]:
    chips: List[str] = []

    confidence = row.get("confidence")
    if pd.notnull(confidence) and confidence >= high_probability_threshold:
        chips.append("High Win Probability")

    if pd.notnull(row.get("prediction_error")) and row["prediction_error"] <= 0.2:
        chips.append("Recent Accuracy")

    return chips


def _format_interval(lower: Optional[float], upper: Optional[float]) -> Optional[str]:
    if pd.notnull(lower) and pd.notnull(upper):
        return f"{lower * 100:0.1f}% – {upper * 100:0.1f}%"
    return None


def _extract_unique_dates(dataset: pd.DataFrame) -> List[pd.Timestamp]:
    if "game_date" not in dataset.columns:
        return []
    valid_dates = pd.to_datetime(dataset["game_date"], errors="coerce").dropna().dt.normalize()
    unique_dates = sorted(valid_dates.unique())
    return [pd.Timestamp(date) for date in unique_dates]


def _extract_unique_teams(dataset: pd.DataFrame) -> List[str]:
    columns = [col for col in ("home_team", "away_team") if col in dataset.columns]
    teams: List[str] = []
    for column in columns:
        teams.extend(dataset[column].dropna().unique().tolist())
    return sorted(set(teams))
