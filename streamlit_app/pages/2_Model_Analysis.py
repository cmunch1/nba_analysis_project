"""
Model Analysis - Technical model diagnostics and monitoring.

This page provides in-depth analysis of model performance for ML practitioners,
including calibration diagnostics, drift monitoring, and error analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components import calibration_charts
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
    .section-divider {
        margin-top: 3rem;
        margin-bottom: 2rem;
        border-top: 2px solid #e0e2e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def main() -> None:
    st.title("Model Analysis")
    st.caption(
        "Technical diagnostics for model evaluation: calibration, drift monitoring, and error analysis. "
        "This page is designed for ML practitioners and technical audiences."
    )

    dataset = load_latest_dataset()
    if dataset.empty:
        st.info("No dashboard data available yet.")
        return

    # Extract different sections from dashboard data
    if 'section' in dataset.columns:
        results_dataset = dataset[dataset['section'] == 'results'].copy()
        drift_dataset = dataset[dataset['section'] == 'drift'].copy()
    else:
        results_dataset = dataset
        drift_dataset = pd.DataFrame()

    if results_dataset.empty:
        st.warning("Model analysis requires historical results data.")
        return

    # Prepare data
    probability_column = _detect_probability_column(results_dataset)
    if probability_column is None:
        st.warning("No probability column found in results data.")
        return

    results_dataset["predicted_probability"] = pd.to_numeric(
        results_dataset[probability_column], errors="coerce"
    )
    results_dataset["game_date"] = pd.to_datetime(results_dataset.get("game_date"), errors="coerce")

    # Use pre-calculated correctness
    if "prediction_correct" in results_dataset.columns:
        results_dataset["correct"] = results_dataset["prediction_correct"].astype(bool)
    elif "predicted_winner_won" in results_dataset.columns:
        results_dataset["correct"] = results_dataset["predicted_winner_won"].astype(bool)
    else:
        st.warning("No prediction correctness data available.")
        return

    # Use pre-calculated actual outcome
    if "predicted_winner_won" in results_dataset.columns:
        results_dataset["actual_outcome"] = results_dataset["predicted_winner_won"].astype(float)
    elif "actual_home_win" in results_dataset.columns:
        results_dataset["actual_outcome"] = results_dataset["actual_home_win"].astype(float)
    else:
        st.warning("No actual outcome data available.")
        return

    # Filter out missing data
    results_dataset = results_dataset.dropna(
        subset=["predicted_probability", "correct", "actual_outcome", "game_date"]
    )

    # Section navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div style="text-align: center; padding: 10px; background-color: #f0f2f6; '
            'border-radius: 5px; margin: 5px;">'
            '📊 <a href="#calibration-diagnostics" style="text-decoration: none; color: #0068c9; '
            'font-weight: 500;">Calibration Diagnostics</a></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div style="text-align: center; padding: 10px; background-color: #f0f2f6; '
            'border-radius: 5px; margin: 5px;">'
            '📈 <a href="#drift-monitoring" style="text-decoration: none; color: #0068c9; '
            'font-weight: 500;">Drift Monitoring</a></div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div style="text-align: center; padding: 10px; background-color: #f0f2f6; '
            'border-radius: 5px; margin: 5px;">'
            '🔍 <a href="#error-analysis" style="text-decoration: none; color: #0068c9; '
            'font-weight: 500;">Error Analysis</a></div>',
            unsafe_allow_html=True
        )
    st.markdown("---")

    # Section 1: Calibration Diagnostics
    st.markdown('<div id="calibration-diagnostics"></div>', unsafe_allow_html=True)
    _render_calibration_diagnostics(results_dataset, probability_column)

    # Section 2: Drift Monitoring
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="drift-monitoring"></div>', unsafe_allow_html=True)
    _render_drift_monitoring(results_dataset, drift_dataset)

    # Section 3: Error Analysis (placeholder for now)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="error-analysis"></div>', unsafe_allow_html=True)
    _render_error_analysis(results_dataset)


def _detect_probability_column(dataset: pd.DataFrame) -> Optional[str]:
    """Detect which probability column to use."""
    for column in [
        "predicted_winner_prob",
        "calibrated_home_win_prob",
        "home_win_probability",
        "home_win_prob",
    ]:
        if column in dataset.columns:
            return column
    return None


def _apply_time_window(frame: pd.DataFrame, days_back: Optional[int]) -> pd.DataFrame:
    """Filter dataframe to a time window."""
    if days_back is None or frame.empty:
        return frame

    date_col = 'date' if 'date' in frame.columns and 'game_date' not in frame.columns else 'game_date'

    if date_col not in frame.columns:
        return frame

    date_series = pd.to_datetime(frame[date_col], errors='coerce')
    max_date = date_series.max()

    if pd.isna(max_date):
        return frame

    cutoff = max_date - pd.Timedelta(days=days_back)
    return frame.loc[date_series >= cutoff]


def _render_calibration_diagnostics(frame: pd.DataFrame, probability_column: str) -> None:
    """Render calibration analysis section."""
    st.subheader("📊 Calibration Diagnostics")

    is_normalized = probability_column == "predicted_winner_prob"
    prob_description = "predicted winner's probability" if is_normalized else "home team win probability"

    with st.expander("ℹ️ What is calibration?", expanded=False):
        st.markdown(
            f"""
            **Calibration** measures whether predicted probabilities match observed frequencies.
            A well-calibrated model's predictions can be trusted as true probabilities.

            - **Perfect calibration**: When the model predicts 70% probability, the outcome occurs ~70% of the time
            - **Poor calibration**: Systematic over- or under-confidence in predictions
            - **Brier Score**: Measures both calibration and sharpness (lower is better, range 0-1)

            **Why it matters**: Calibrated probabilities enable better decision-making under uncertainty.
            For example, betting strategies require accurate probability estimates, not just correct predictions.

            This chart shows where the model places its confidence (blue bars) and how well those
            confidence levels match reality (blue line vs gray diagonal = perfect calibration).
            """
        )

    if frame.empty:
        st.info("Not enough games to display calibration analysis.")
        return

    # Time period filter
    time_filter = st.radio(
        "Time Period",
        ["Last 14 days", "Last 30 days", "Entire Season"],
        index=2,
        horizontal=True,
        key="calib_filter"
    )

    filter_map = {
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Entire Season": None
    }
    days_back = filter_map[time_filter]

    filtered_frame = _apply_time_window(frame, days_back)

    if filtered_frame.empty:
        st.info("Not enough games for the selected period.")
        return

    # Prepare binned data
    bin_stats = calibration_charts.prepare_calibration_bins(
        df=filtered_frame,
        probability_col=probability_column,
        outcome_col="actual_outcome",
        n_bins=20,
        min_prob=0.5,
    )

    if bin_stats.empty:
        st.info("Not enough predictions in 50%+ probability range.")
        return

    # Display overall Brier score
    overall_brier = calibration_charts.calculate_brier_score(
        df=filtered_frame,
        probability_col=probability_column,
        outcome_col="actual_outcome",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Brier Score", f"{overall_brier:.4f}")
    with col2:
        avg_prob = filtered_frame[probability_column].mean()
        st.metric("Average Predicted Probability", f"{avg_prob:.1%}")
    with col3:
        actual_rate = filtered_frame["actual_outcome"].mean()
        st.metric("Actual Win Rate", f"{actual_rate:.1%}")

    # Create and display chart
    fig = calibration_charts.create_calibration_bin_chart(bin_stats)
    st.plotly_chart(fig, use_container_width=True)

    # Display bin-level details
    with st.expander("📋 View bin-level statistics", expanded=False):
        display_df = bin_stats[["bin_label", "count", "avg_predicted", "actual_rate", "brier_score"]].copy()
        display_df.columns = ["Probability Bin", "Games", "Avg Predicted", "Actual Rate", "Brier Score"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_drift_monitoring(results_frame: pd.DataFrame, drift_frame: pd.DataFrame) -> None:
    """Render drift monitoring section."""
    st.subheader("📈 Model Drift Monitoring")

    with st.expander("ℹ️ What is model drift?", expanded=False):
        st.markdown(
            """
            **Model drift** occurs when model performance degrades over time due to changes in:
            - **Concept drift**: The relationship between features and outcomes changes
            - **Data drift**: The distribution of input features changes
            - **Prediction drift**: The model's predictions change even for similar inputs

            **Why monitor drift**:
            - Detect when the model needs retraining
            - Identify seasonal or temporal patterns
            - Catch data quality issues early
            - Ensure production reliability

            This section tracks model accuracy and calibration metrics over time to detect drift.
            """
        )

    if results_frame.empty:
        st.info("No results data available for drift analysis.")
        return

    # Time period filter
    time_filter = st.radio(
        "Time Period",
        ["Last 14 days", "Last 30 days", "Entire Season"],
        index=2,
        horizontal=True,
        key="drift_filter"
    )

    filter_map = {
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Entire Season": None
    }
    days_back = filter_map[time_filter]

    filtered_results = _apply_time_window(results_frame, days_back)

    if filtered_results.empty:
        st.info("No data available for the selected period.")
        return

    # Use drift data if available, otherwise calculate from results
    if not drift_frame.empty and 'date' in drift_frame.columns:
        filtered_drift = _apply_time_window(drift_frame, days_back)

        if not filtered_drift.empty:
            drift_data = filtered_drift.copy()
            drift_data['game_date'] = pd.to_datetime(drift_data['date'])
            drift_data['accuracy'] = pd.to_numeric(drift_data['accuracy'], errors='coerce')
            drift_data['brier_score'] = pd.to_numeric(drift_data.get('brier_score'), errors='coerce')
            drift_data['calibration_error'] = pd.to_numeric(drift_data.get('calibration_error'), errors='coerce')
            drift_data = drift_data.sort_values('game_date')
        else:
            drift_data = None
    else:
        drift_data = None

    # If no drift data, calculate from results
    if drift_data is None:
        daily_metrics = (
            filtered_results.groupby("game_date")
            .agg(
                accuracy=("correct", "mean"),
                games=("correct", "size"),
            )
            .reset_index()
            .sort_values("game_date")
        )
        drift_data = daily_metrics

    if drift_data.empty:
        st.info("Unable to compute drift metrics.")
        return

    # Create drift visualization
    fig = go.Figure()

    # Accuracy over time
    fig.add_trace(go.Scatter(
        x=drift_data['game_date'],
        y=drift_data['accuracy'],
        mode='lines+markers',
        name='Daily Accuracy',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Add rolling average if enough data points
    if len(drift_data) >= 7:
        rolling_acc = drift_data['accuracy'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=drift_data['game_date'],
            y=rolling_acc,
            mode='lines',
            name='7-Day Rolling Average',
            line=dict(color='red', width=2, dash='dash')
        ))

    fig.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display additional drift metrics if available
    if 'brier_score' in drift_data.columns and drift_data['brier_score'].notna().any():
        col1, col2 = st.columns(2)

        with col1:
            fig_brier = go.Figure()
            fig_brier.add_trace(go.Scatter(
                x=drift_data['game_date'],
                y=drift_data['brier_score'],
                mode='lines+markers',
                name='Brier Score',
                line=dict(color='purple', width=2),
                marker=dict(size=6)
            ))
            fig_brier.update_layout(
                title="Brier Score Over Time",
                xaxis_title="Date",
                yaxis_title="Brier Score (lower is better)",
                height=300
            )
            st.plotly_chart(fig_brier, use_container_width=True)

        with col2:
            if 'calibration_error' in drift_data.columns and drift_data['calibration_error'].notna().any():
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=drift_data['game_date'],
                    y=drift_data['calibration_error'],
                    mode='lines+markers',
                    name='Calibration Error',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6)
                ))
                fig_cal.update_layout(
                    title="Calibration Error Over Time",
                    xaxis_title="Date",
                    yaxis_title="Expected Calibration Error",
                    height=300
                )
                st.plotly_chart(fig_cal, use_container_width=True)

    # Summary statistics
    st.subheader("Drift Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mean_acc = drift_data['accuracy'].mean()
        st.metric("Mean Accuracy", f"{mean_acc:.1%}")

    with col2:
        std_acc = drift_data['accuracy'].std()
        st.metric("Accuracy Std Dev", f"{std_acc:.1%}")

    with col3:
        min_acc = drift_data['accuracy'].min()
        st.metric("Worst Day", f"{min_acc:.1%}")

    with col4:
        max_acc = drift_data['accuracy'].max()
        st.metric("Best Day", f"{max_acc:.1%}")


def _render_error_analysis(frame: pd.DataFrame) -> None:
    """Render error analysis section (placeholder)."""
    st.subheader("🔍 Error Analysis")

    with st.expander("ℹ️ What is error analysis?", expanded=False):
        st.markdown(
            """
            **Error analysis** examines patterns in model failures to identify:
            - Systematic biases (e.g., better at predicting home wins)
            - Failure modes (e.g., struggles with close games)
            - Data quality issues
            - Opportunities for improvement

            This section will include:
            - Error rate by confidence level
            - Home vs away prediction accuracy
            - Error patterns by team
            - High-confidence failure cases
            """
        )

    st.info("🚧 Error analysis coming soon! This section will include detailed breakdowns of model failures and systematic biases.")

    # Quick summary for now
    if not frame.empty:
        col1, col2, col3 = st.columns(3)

        with col1:
            total_errors = (~frame['correct']).sum()
            error_rate = (~frame['correct']).mean()
            st.metric("Total Errors", f"{int(total_errors)}", f"{error_rate:.1%} error rate")

        with col2:
            high_conf_errors = frame[
                (frame['predicted_probability'] >= 0.70) & (~frame['correct'])
            ]
            st.metric("High Confidence Errors (70%+)", len(high_conf_errors))

        with col3:
            low_conf_errors = frame[
                (frame['predicted_probability'] < 0.60) & (~frame['correct'])
            ]
            st.metric("Low Confidence Errors (<60%)", len(low_conf_errors))


if __name__ == "__main__":
    main()
