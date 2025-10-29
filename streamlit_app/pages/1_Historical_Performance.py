from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.data_loader import load_latest_dataset


WINDOW_OPTIONS = {
    "Entire season": None,
    "Last 30 days": 30,
    "Last 7 days": 7,
}


@dataclass
class EvaluationFrame:
    """Standardised view of prediction outcomes."""

    frame: pd.DataFrame
    probability_column: str


def main() -> None:
    st.title("Historical Performance")
    st.caption("Assess calibration, accuracy, and drift across recent prediction history.")

    dataset = load_latest_dataset()
    if dataset.empty:
        st.info("No dashboard data available yet.")
        return

    evaluation = _prepare_evaluation_frame(dataset)
    if evaluation.frame.empty:
        st.warning(
            "Historical metrics require actual results and calibrated probabilities."
            " Update the dashboard prep pipeline to include these columns."
        )
        return

    with st.sidebar:
        st.subheader("Historical Filters")
        selection = st.selectbox("Time window", list(WINDOW_OPTIONS.keys()))

    filtered = _apply_time_window(evaluation.frame, WINDOW_OPTIONS[selection])

    _render_calibration_chart(filtered, evaluation.probability_column)
    _render_daily_accuracy(filtered)
    _render_summary_table(filtered)


def _prepare_evaluation_frame(dataset: pd.DataFrame) -> EvaluationFrame:
    probability_column = _detect_probability_column(dataset)
    if probability_column is None:
        return EvaluationFrame(frame=pd.DataFrame(), probability_column="calibrated_home_win_prob")

    frame = dataset.copy()
    frame["predicted_probability"] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["actual_outcome"] = _derive_actual_outcome(frame)
    frame["game_date"] = pd.to_datetime(frame.get("game_date"), errors="coerce")
    frame["win_probability"] = pd.to_numeric(frame.get("confidence"), errors="coerce")
    frame["prediction_error"] = pd.to_numeric(frame.get("prediction_error"), errors="coerce")

    frame = frame.dropna(subset=["predicted_probability", "actual_outcome", "game_date"])
    frame["predicted_home_win"] = frame["predicted_probability"] >= 0.5
    frame["correct"] = (
        frame["predicted_home_win"].astype(float) == frame["actual_outcome"].astype(float)
    )

    return EvaluationFrame(frame=frame, probability_column="predicted_probability")


def _apply_time_window(frame: pd.DataFrame, days_back: Optional[int]) -> pd.DataFrame:
    if days_back is None or frame.empty:
        return frame

    max_date = frame["game_date"].max()
    if pd.isna(max_date):
        return frame

    cutoff = max_date - pd.Timedelta(days=days_back)
    return frame.loc[frame["game_date"] >= cutoff]


def _render_calibration_chart(frame: pd.DataFrame, probability_column: str) -> None:
    st.subheader("Calibration Curve")
    if frame.empty:
        st.info("Not enough games with completed results to display calibration.")
        return

    bins = pd.cut(
        frame[probability_column],
        bins=np.linspace(0.0, 1.0, num=11),
        include_lowest=True,
    )
    calibration = (
        frame.assign(prob_bin=bins)
        .groupby("prob_bin")
        .agg(
            predicted_prob=(probability_column, "mean"),
            actual_rate=("actual_outcome", "mean"),
            count=(probability_column, "size"),
        )
        .dropna()
    )
    calibration = calibration[calibration["count"] >= 10]
    if calibration.empty:
        st.info("Not enough samples per probability bin yet.")
        return

    fig = px.scatter(
        calibration,
        x="predicted_prob",
        y="actual_rate",
        size="count",
        hover_data={"count": True},
        labels={
            "predicted_prob": "Predicted home win probability",
            "actual_rate": "Observed win rate",
        },
        title="Model calibration",
    )
    perfect_line = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(color="gray", dash="dash"),
        showlegend=True,
    )
    fig.add_trace(perfect_line)
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_daily_accuracy(frame: pd.DataFrame) -> None:
    st.subheader("Daily Performance")
    if frame.empty:
        st.info("No games available for the selected period.")
        return

    daily = (
        frame.groupby("game_date")
        .agg(
            accuracy=("correct", "mean"),
            avg_win_probability=("win_probability", "mean"),
            avg_interval=("prediction_error", "mean"),
            games=("correct", "size"),
        )
        .reset_index()
    )
    if daily.empty:
        st.info("Unable to compute daily metrics for the selected period.")
        return

    fig = px.line(
        daily.sort_values("game_date"),
        x="game_date",
        y="accuracy",
        markers=True,
        labels={"game_date": "Game date", "accuracy": "Accuracy"},
    )
    fig.update_traces(mode="lines+markers")
    fig.update_yaxis(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Daily metric details"):
        st.dataframe(daily, use_container_width=True)


def _render_summary_table(frame: pd.DataFrame) -> None:
    st.subheader("Metric Snapshot")
    if frame.empty:
        st.info("No summary metrics available.")
        return

    summary = {
        "Games evaluated": len(frame),
        "Accuracy": f"{frame['correct'].mean() * 100:0.1f}%",
    }

    if frame["win_probability"].notna().any():
        summary["Avg win probability"] = f"{frame['win_probability'].mean() * 100:0.1f}%"
    if frame["prediction_error"].notna().any():
        summary["Avg prediction error"] = f"{frame['prediction_error'].mean():0.3f}"

    st.write(summary)


def _detect_probability_column(dataset: pd.DataFrame) -> Optional[str]:
    for column in [
        "calibrated_home_win_prob",
        "home_win_probability",
        "home_win_prob",
    ]:
        if column in dataset.columns:
            return column
    return None


def _derive_actual_outcome(dataset: pd.DataFrame) -> pd.Series:
    if {"actual_home_score", "actual_away_score"}.issubset(dataset.columns):
        return (dataset["actual_home_score"] > dataset["actual_away_score"]).astype(float)

    if {"actual_winner", "home_team"}.issubset(dataset.columns):
        return (dataset["actual_winner"] == dataset["home_team"]).astype(float)

    if "prediction_correct" in dataset.columns:
        return dataset["prediction_correct"].astype(float)

    return pd.Series([np.nan] * len(dataset), index=dataset.index)


if __name__ == "__main__":
    main()
