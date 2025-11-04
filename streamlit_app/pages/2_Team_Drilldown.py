from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.data_loader import load_latest_dataset

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


def main() -> None:
    st.title("Team Drilldown")
    st.caption("Inspect model win probabilities and outcomes for a single team.")

    dataset = load_latest_dataset()
    if dataset.empty:
        st.info("No dashboard data available yet.")
        return

    teams = _extract_teams(dataset)
    if not teams:
        st.warning("Dashboard dataset does not include team identifiers.")
        return

    with st.sidebar:
        st.subheader("Team Selection")
        selected_team = st.selectbox("Team", teams)
        max_rows = st.slider("Recent games to display", min_value=5, max_value=50, value=15, step=5)

    team_frame = _prepare_team_frame(dataset, selected_team)
    if team_frame.empty:
        st.info("No games recorded for the selected team yet.")
        return

    _render_team_metrics(team_frame)
    _render_probability_timeline(team_frame)
    _render_home_away_breakdown(team_frame)
    _render_recent_games_table(team_frame, max_rows)


def _extract_teams(dataset: pd.DataFrame) -> List[str]:
    columns = [col for col in ("home_team", "away_team") if col in dataset.columns]
    if not columns:
        return []
    teams: List[str] = []
    for column in columns:
        teams.extend(dataset[column].dropna().unique().tolist())
    return sorted(set(teams))


def _prepare_team_frame(dataset: pd.DataFrame, team: str) -> pd.DataFrame:
    mask = False
    if "home_team" in dataset.columns:
        mask = mask | (dataset["home_team"] == team)
    if "away_team" in dataset.columns:
        mask = mask | (dataset["away_team"] == team)

    team_frame = dataset.loc[mask].copy()
    if team_frame.empty:
        return team_frame

    team_frame["game_date"] = pd.to_datetime(team_frame.get("game_date"), errors="coerce")
    team_frame["is_home"] = team_frame.get("home_team") == team
    team_frame["location"] = np.where(team_frame["is_home"], "Home", "Away")
    team_frame["opponent"] = np.where(
        team_frame["is_home"],
        team_frame.get("away_team"),
        team_frame.get("home_team"),
    )

    win_prob_column = _detect_probability_column(team_frame)
    if win_prob_column:
        team_frame["team_win_probability"] = np.where(
            team_frame["is_home"],
            pd.to_numeric(team_frame[win_prob_column], errors="coerce"),
            1.0 - pd.to_numeric(team_frame[win_prob_column], errors="coerce"),
        )

    team_frame["win_probability"] = pd.to_numeric(team_frame.get("confidence"), errors="coerce")
    team_frame["prediction_error"] = pd.to_numeric(team_frame.get("prediction_error"), errors="coerce")

    if "predicted_winner" in team_frame.columns:
        team_frame["team_predicted_win"] = team_frame["predicted_winner"] == team
    elif "team_win_probability" in team_frame.columns:
        team_frame["team_predicted_win"] = team_frame["team_win_probability"] >= 0.5

    if "actual_winner" in team_frame.columns:
        team_frame["team_actual_win"] = team_frame["actual_winner"] == team
    elif {"actual_home_score", "actual_away_score"}.issubset(team_frame.columns):
        is_home_win = team_frame["actual_home_score"] > team_frame["actual_away_score"]
        team_frame["team_actual_win"] = np.where(team_frame["is_home"], is_home_win, ~is_home_win)
    elif "prediction_correct" in team_frame.columns and "team_predicted_win" in team_frame.columns:
        correctness = team_frame["prediction_correct"].astype("boolean")
        team_frame["team_actual_win"] = np.where(
            team_frame["team_predicted_win"],
            correctness,
            ~correctness,
        )

    if "team_predicted_win" in team_frame.columns:
        team_frame["team_predicted_win"] = team_frame["team_predicted_win"].astype("boolean")

    if "team_actual_win" in team_frame.columns:
        team_frame["team_actual_win"] = team_frame["team_actual_win"].astype("boolean")

    return team_frame


def _render_team_metrics(team_frame: pd.DataFrame) -> None:
    st.subheader("Team Metrics")

    games_played = len(team_frame)
    games_with_results = team_frame["team_actual_win"].notna().sum() if "team_actual_win" in team_frame else 0
    accuracy = None

    if "team_actual_win" in team_frame and "team_predicted_win" in team_frame:
        evaluated = team_frame.dropna(subset=["team_actual_win", "team_predicted_win"])
        if not evaluated.empty:
            accuracy = float((evaluated["team_actual_win"] == evaluated["team_predicted_win"]).mean())

    avg_win_probability = team_frame["win_probability"].mean() if "win_probability" in team_frame else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Games tracked", games_played)
    col2.metric("Games completed", games_with_results)
    if accuracy is not None:
        col3.metric("Prediction accuracy", f"{accuracy * 100:0.1f}%")
    else:
        col3.metric("Prediction accuracy", "N/A")

    if not np.isnan(avg_win_probability):
        st.caption(f"Average win probability: {avg_win_probability * 100:0.1f}%")


def _render_probability_timeline(team_frame: pd.DataFrame) -> None:
    st.subheader("Prediction Timeline")

    if "team_win_probability" not in team_frame.columns or team_frame["game_date"].isna().all():
        st.info("Win probabilities or game dates are missing for this team.")
        return

    timeline = team_frame.dropna(subset=["team_win_probability", "game_date"]).sort_values("game_date")
    if timeline.empty:
        st.info("No valid probabilities available yet.")
        return

    result_map = {True: "Win", False: "Loss"}
    timeline["result_label"] = timeline.get("team_actual_win").map(result_map).fillna("Pending")

    fig = px.line(
        timeline,
        x="game_date",
        y="team_win_probability",
        color="location",
        markers=True,
        hover_data={
            "opponent": True,
            "team_win_probability": ":.2f",
            "result_label": True,
        },
        labels={"team_win_probability": "Win probability"},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_home_away_breakdown(team_frame: pd.DataFrame) -> None:
    st.subheader("Home vs Away Performance")

    if "team_actual_win" not in team_frame or "team_predicted_win" not in team_frame:
        st.info("Insufficient data to compute home/away accuracy.")
        return

    evaluated = team_frame.dropna(subset=["team_actual_win", "team_predicted_win"])
    if evaluated.empty:
        st.info("No completed games for this team yet.")
        return

    accuracy_by_location = (
        evaluated.groupby("location")
        .apply(lambda df: float((df["team_actual_win"] == df["team_predicted_win"]).mean()))
        .reset_index(name="accuracy")
    )

    fig = px.bar(
        accuracy_by_location,
        x="location",
        y="accuracy",
        range_y=[0, 1],
        text="accuracy",
        labels={"accuracy": "Accuracy"},
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def _render_recent_games_table(team_frame: pd.DataFrame, max_rows: int) -> None:
    st.subheader("Recent Games")

    columns = [
        "game_date",
        "location",
        "opponent",
        "team_win_probability",
        "predicted_winner",
        "actual_winner",
        "win_probability",
        "prediction_error",
    ]
    available_columns = [col for col in columns if col in team_frame.columns]

    recent_games = team_frame.sort_values("game_date", ascending=False).head(max_rows)
    if recent_games.empty:
        st.info("No games available to display.")
        return

    st.dataframe(
        recent_games[available_columns],
        use_container_width=True,
    )


def _detect_probability_column(dataset: pd.DataFrame) -> Optional[str]:
    for column in [
        "calibrated_home_win_prob",
        "home_win_probability",
        "home_win_prob",
    ]:
        if column in dataset.columns:
            return column
    return None


if __name__ == "__main__":
    main()
