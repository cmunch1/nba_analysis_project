from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


def render_kpi_panel(
    dataset: pd.DataFrame,
    high_probability_threshold: float,
) -> None:
    """
    Display headline KPIs for the active matchup selection.
    """
    st.subheader("Matchup Summary")

    if dataset.empty:
        st.info("No games available for the current selection.")
        return

    total_games = len(dataset)
    confidence_series = dataset["confidence"] if "confidence" in dataset.columns else pd.Series(dtype=float)
    high_prob_games = (
        int((confidence_series >= high_probability_threshold).sum()) if not confidence_series.empty else 0
    )
    avg_win_probability = confidence_series.mean() if not confidence_series.empty else np.nan

    accuracy = _compute_accuracy(dataset)

    col1, col2, col3 = st.columns(3)
    col1.metric("Games", total_games)
    col2.metric(
        "High-Probability Picks",
        high_prob_games,
        help=f"Win Probability ≥ {high_probability_threshold:.2f}",
    )
    if not np.isnan(avg_win_probability):
        col3.metric("Avg Win Probability", f"{avg_win_probability * 100:0.1f}%")
    else:
        col3.metric("Avg Win Probability", "N/A")

    if accuracy is not None:
        st.caption(f"Recent accuracy for filtered games: {accuracy * 100:0.1f}%")


def _compute_accuracy(dataset: pd.DataFrame) -> Optional[float]:
    """
    Estimate accuracy using available columns.
    """
    if "prediction_correct" in dataset.columns:
        correctness = dataset["prediction_correct"].dropna()
        if correctness.empty:
            return None
        return float(correctness.mean())

    if {"predicted_winner", "actual_winner"}.issubset(dataset.columns):
        comparisons = dataset.dropna(subset=["predicted_winner", "actual_winner"])
        if comparisons.empty:
            return None
        correct = (comparisons["predicted_winner"] == comparisons["actual_winner"]).mean()
        return float(correct)

    return None
