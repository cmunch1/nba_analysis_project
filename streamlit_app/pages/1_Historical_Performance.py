from __future__ import annotations

from dataclasses import dataclass
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
    </style>
    """,
    unsafe_allow_html=True
)

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

    # Extract different sections from dashboard data
    if 'section' in dataset.columns:
        results_dataset = dataset[dataset['section'] == 'results'].copy()
        drift_dataset = dataset[dataset['section'] == 'drift'].copy()
    else:
        results_dataset = dataset
        drift_dataset = pd.DataFrame()

    evaluation = _prepare_evaluation_frame(results_dataset)
    if evaluation.frame.empty:
        st.warning(
            "Historical metrics require actual results and calibrated probabilities."
            " Update the dashboard prep pipeline to include these columns."
        )
        return

    # No global filter - each chart section will have its own filter
    results_frame = evaluation.frame
    drift_frame = drift_dataset

    # Section navigation with styled buttons
    st.markdown("---")
    st.markdown(
        """
        <style>
        .nav-button {
            text-align: center;
            padding: 10px;
            background-color: #f0f2f6;
            border-radius: 5px;
            margin: 5px;
        }
        .nav-button a {
            text-decoration: none;
            color: #0068c9;
            font-weight: 500;
        }
        .nav-button:hover {
            background-color: #e0e2e6;
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="nav-button">📊 <a href="#daily-performance">Daily Performance</a></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="nav-button">🎯 <a href="#threshold-analysis">Threshold Analysis</a></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="nav-button">🏀 <a href="#team-accuracy">Team Accuracy</a></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="nav-button">📋 <a href="#recent-games">Recent Games</a></div>', unsafe_allow_html=True)
    st.markdown("---")

    # Link to Model Analysis page for technical details
    st.info("🔬 **For technical users**: View [Model Analysis](/Model_Analysis) for calibration diagnostics, drift monitoring, and error analysis.")

    # Section 1: Daily Performance
    st.markdown('<div id="daily-performance"></div>', unsafe_allow_html=True)
    _render_daily_accuracy(results_frame, drift_frame)

    # Section 2: Threshold Analysis
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="threshold-analysis"></div>', unsafe_allow_html=True)
    _render_threshold_analysis(results_frame, evaluation.probability_column)

    # Section 3: Team-Level Accuracy
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="team-accuracy"></div>', unsafe_allow_html=True)
    _render_team_accuracy_analysis(results_frame)

    # Section 4: Recent Games Detail
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="recent-games"></div>', unsafe_allow_html=True)
    _render_recent_games_table(results_frame)

    # _render_summary_table(filtered)


def _prepare_evaluation_frame(dataset: pd.DataFrame) -> EvaluationFrame:
    """
    Prepare evaluation frame from dashboard data.

    This function now only performs type conversion and basic cleaning.
    All calculations (correct, probability, outcome) are pre-calculated by the dashboard prep pipeline.
    """
    probability_column = _detect_probability_column(dataset)
    if probability_column is None:
        return EvaluationFrame(frame=pd.DataFrame(), probability_column="predicted_winner_prob")

    frame = dataset.copy()

    # Type conversions only - no calculations
    frame["predicted_probability"] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["game_date"] = pd.to_datetime(frame.get("game_date"), errors="coerce")
    frame["confidence"] = pd.to_numeric(frame.get("confidence"), errors="coerce")
    frame["prediction_error"] = pd.to_numeric(frame.get("prediction_error"), errors="coerce")

    # Use pre-calculated prediction_correct from dashboard prep pipeline
    # This is the authoritative source calculated by results_aggregator.py
    if "prediction_correct" in frame.columns:
        frame["correct"] = frame["prediction_correct"].astype(bool)
    elif "predicted_winner_won" in frame.columns:
        # Fallback to alternative pre-calculated column
        frame["correct"] = frame["predicted_winner_won"].astype(bool)
    else:
        # No pre-calculated correctness available
        return EvaluationFrame(frame=pd.DataFrame(), probability_column=probability_column)

    # Use pre-calculated actual outcome (for calibration charts that need binary outcome)
    if "predicted_winner_won" in frame.columns:
        # For normalized probabilities, actual_outcome is whether predicted winner won
        frame["actual_outcome"] = frame["predicted_winner_won"].astype(float)
    elif "actual_home_win" in frame.columns:
        # For home team probabilities, actual_outcome is whether home team won
        frame["actual_outcome"] = frame["actual_home_win"].astype(float)
    else:
        # Derive from actual_winner if needed
        frame["actual_outcome"] = _derive_actual_outcome(frame)

    # Filter out rows with missing critical data
    required_cols = ["predicted_probability", "correct", "game_date"]
    frame = frame.dropna(subset=required_cols)

    return EvaluationFrame(frame=frame, probability_column=probability_column)


def _apply_time_window(frame: pd.DataFrame, days_back: Optional[int]) -> pd.DataFrame:
    """Filter dataframe to a time window."""
    if days_back is None or frame.empty:
        return frame

    # Determine which date column to use (drift section uses 'date', results use 'game_date')
    date_col = 'date' if 'date' in frame.columns and 'game_date' not in frame.columns else 'game_date'

    if date_col not in frame.columns:
        return frame

    # Convert to datetime if not already
    date_series = pd.to_datetime(frame[date_col], errors='coerce')
    max_date = date_series.max()

    if pd.isna(max_date):
        return frame

    cutoff = max_date - pd.Timedelta(days=days_back)
    return frame.loc[date_series >= cutoff]


def _render_daily_accuracy(frame: pd.DataFrame, drift_data: pd.DataFrame) -> None:
    """
    Render daily accuracy chart using pre-calculated drift data from dashboard prep pipeline.

    Args:
        frame: Full results data
        drift_data: Pre-calculated daily metrics from dashboard prep pipeline
    """
    st.subheader("📊 Daily Performance")

    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(
            """
            This chart tracks model accuracy over time with three key metrics:

            - **Daily Accuracy** (blue): Model's accuracy for each day
            - **Season Running Accuracy** (green): Cumulative accuracy from season start
            - **Period Average** (red dashed): Average accuracy for the selected time period
            - **Season Average** (orange dashed): Overall season accuracy baseline

            Use the time period filter to focus on recent performance or view the entire season.
            """
        )

    if frame.empty:
        st.info("No games available.")
        return

    # Time period filter for this chart
    time_filter = st.radio(
        "Time Period",
        ["Last 7 days", "Last 14 days", "Last 30 days", "Entire Season"],
        index=3,  # Default to Entire Season
        horizontal=True,
        key="daily_perf_filter"
    )

    # Map filter to days
    filter_map = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Entire Season": None
    }
    days_back = filter_map[time_filter]

    # Apply filter
    filtered_frame = _apply_time_window(frame, days_back)
    filtered_drift = _apply_time_window(drift_data, days_back) if not drift_data.empty else pd.DataFrame()

    if filtered_frame.empty:
        st.info("No games available for the selected period.")
        return

    # Calculate season average (from full unfiltered frame)
    season_avg_accuracy = frame["correct"].mean()

    # Use pre-calculated drift data if available, otherwise fall back to calculating from frame
    if not filtered_drift.empty and 'date' in filtered_drift.columns and 'accuracy' in filtered_drift.columns:
        # Use pre-calculated daily accuracy from drift section
        daily = filtered_drift.copy()
        daily['game_date'] = pd.to_datetime(daily['date'])
        daily['accuracy'] = pd.to_numeric(daily['accuracy'], errors='coerce')

        # Calculate games per day from results frame
        games_per_day = filtered_frame.groupby("game_date").size().reset_index(name="games")
        daily = daily.merge(games_per_day, on="game_date", how="left")

    else:
        # Fallback: calculate from results frame
        daily = (
            filtered_frame.groupby("game_date")
            .agg(
                accuracy=("correct", "mean"),
                games=("correct", "size"),
            )
            .reset_index()
        )

    if daily.empty:
        st.info("Unable to compute daily metrics for the selected period.")
        return

    # Sort by date for proper chronological ordering
    daily = daily.sort_values("game_date").reset_index(drop=True)

    # Calculate cumulative season accuracy (always from full season data for accuracy)
    # But only show it for dates in the filtered period
    frame_sorted = frame.sort_values("game_date").copy()
    date_grouped = frame_sorted.groupby("game_date")["correct"].agg(["sum", "count"]).reset_index()
    date_grouped = date_grouped.sort_values("game_date")
    date_grouped["cumsum_correct"] = date_grouped["sum"].cumsum()
    date_grouped["cumsum_total"] = date_grouped["count"].cumsum()
    date_grouped["cumulative_accuracy"] = date_grouped["cumsum_correct"] / date_grouped["cumsum_total"]

    # Only keep cumulative data for dates in the filtered period
    date_grouped_filtered = date_grouped[date_grouped["game_date"].isin(daily["game_date"])].copy()

    # Merge cumulative accuracy to daily dataframe
    daily = daily.merge(
        date_grouped_filtered[["game_date", "cumulative_accuracy"]],
        on="game_date",
        how="inner"  # Use inner join to ensure we only keep dates that exist in both
    )

    # Ensure we're only showing the filtered period
    if not daily.empty and days_back is not None:
        # Double-check: remove any dates outside the filtered period
        max_date = daily["game_date"].max()
        cutoff_date = max_date - pd.Timedelta(days=days_back)
        daily = daily[daily["game_date"] >= cutoff_date]

    # Final check for empty data after filtering
    if daily.empty:
        st.info("No daily data available for the selected period.")
        return

    # Calculate average accuracy for the selected period
    period_avg_accuracy = filtered_frame["correct"].mean()

    # Create the figure with daily accuracy
    fig = go.Figure()

    # Daily accuracy line
    fig.add_trace(go.Scatter(
        x=daily["game_date"],
        y=daily["accuracy"],
        mode="lines+markers",
        name="Daily Accuracy",
        line=dict(color="blue"),
        marker=dict(size=8)
    ))

    # Cumulative season accuracy line
    fig.add_trace(go.Scatter(
        x=daily["game_date"],
        y=daily["cumulative_accuracy"],
        mode="lines",
        name="Season Running Accuracy",
        line=dict(color="green", width=2)
    ))

    # Period average (horizontal line)
    fig.add_trace(go.Scatter(
        x=[daily["game_date"].min(), daily["game_date"].max()],
        y=[period_avg_accuracy, period_avg_accuracy],
        mode="lines",
        name=f"Period Average ({period_avg_accuracy*100:.1f}%)",
        line=dict(color="red", dash="dash", width=2)
    ))

    # Season average baseline (horizontal line)
    fig.add_trace(go.Scatter(
        x=[daily["game_date"].min(), daily["game_date"].max()],
        y=[season_avg_accuracy, season_avg_accuracy],
        mode="lines",
        name=f"Season Average ({season_avg_accuracy*100:.1f}%)",
        line=dict(color="orange", dash="dash", width=2)
    ))

    fig.update_layout(
        xaxis_title="Game Date",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # with st.expander("Daily metric details"):
    #     st.dataframe(daily, use_container_width=True)


def _render_threshold_analysis(frame: pd.DataFrame, probability_column: str) -> None:
    """Render error rate vs confidence threshold analysis."""
    st.subheader("🎯 Error Rate vs Confidence Threshold")

    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(
            """
            This chart answers: **"If I only act on higher-confidence predictions, what error rate do I get?"**

            - **Red line**: Error rate (1 - accuracy) for predictions at or above each threshold
            - **Gray area**: Number of games available at each threshold
            - **Vertical lines**: Common confidence levels (60%, 70%, 80%)

            **How to use**: If you want to maximize accuracy, find the threshold where the red line
            is lowest while still having enough games (gray area) to be statistically meaningful.
            """
        )

    if frame.empty:
        st.info("Not enough games to display threshold analysis.")
        return

    # Time period filter for this chart
    time_filter = st.radio(
        "Time Period",
        ["Last 14 days", "Last 30 days", "Entire Season"],
        index=2,  # Default to Entire Season
        horizontal=True,
        key="threshold_filter"
    )

    # Map filter to days
    filter_map = {
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Entire Season": None
    }
    days_back = filter_map[time_filter]

    # Apply filter
    filtered_frame = _apply_time_window(frame, days_back)

    if filtered_frame.empty:
        st.info("Not enough games to display threshold analysis for the selected period.")
        return

    # Calculate threshold metrics
    threshold_stats = calibration_charts.calculate_threshold_metrics(
        df=filtered_frame,
        probability_col=probability_column,
        outcome_col="actual_outcome",
        correct_col="correct",
        min_threshold=0.5,
        max_threshold=1.0,
        step=0.01,
    )

    if threshold_stats.empty:
        st.info("Unable to calculate threshold metrics.")
        return

    # Create and display chart
    fig = calibration_charts.create_threshold_error_chart(threshold_stats)
    st.plotly_chart(fig, use_container_width=True)

    # Display summary table for key thresholds
    st.subheader("Key Threshold Summary")
    summary_table = calibration_charts.create_threshold_summary_table(threshold_stats)

    if not summary_table.empty:
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
    else:
        st.info("Unable to generate threshold summary.")


def _render_team_accuracy_analysis(frame: pd.DataFrame) -> None:
    """Render team-level accuracy analysis with bar chart."""
    st.subheader("🏀 Team-Level Accuracy")

    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(
            """
            This chart shows **how accurate the model is for each team**.

            - Teams are ranked from **worst accuracy** (bottom) to **best accuracy** (top)
            - Bar length represents accuracy percentage
            - Helps identify which teams the model predicts well vs struggles with

            **Note:** Only teams with at least 5 predictions in the selected time window are shown.
            """
        )

    if frame.empty:
        st.info("No games available for team analysis.")
        return

    # Time period filter for this chart
    time_filter = st.radio(
        "Time Period",
        ["Last 14 days", "Last 30 days", "Entire Season"],
        index=2,  # Default to Entire Season
        horizontal=True,
        key="team_accuracy_filter"
    )

    # Map filter to days
    filter_map = {
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Entire Season": None
    }
    days_back = filter_map[time_filter]

    # Apply filter
    filtered_frame = _apply_time_window(frame, days_back)

    if filtered_frame.empty:
        st.info("No games available for team analysis in the selected period.")
        return

    # Combine home and away games for each team
    team_stats = []

    for team in pd.concat([filtered_frame['home_team'], filtered_frame['away_team']]).unique():
        if pd.isna(team):
            continue

        # Get games where this team played
        team_games = filtered_frame[
            (filtered_frame['home_team'] == team) | (filtered_frame['away_team'] == team)
        ].copy()

        if len(team_games) < 5:  # Skip teams with too few games
            continue

        # Calculate team-specific accuracy
        accuracy = team_games['correct'].mean()

        team_stats.append({
            'Team': team,
            'Games': len(team_games),
            'Accuracy': accuracy,
            'Correct': int(team_games['correct'].sum()),
        })

    if not team_stats:
        st.info("Not enough games per team (minimum 5 required).")
        return

    team_df = pd.DataFrame(team_stats)
    team_df = team_df.sort_values('Accuracy', ascending=True)  # Ascending for horizontal bar chart

    # Create horizontal bar chart
    fig = go.Figure()

    # Color bars based on accuracy
    colors = team_df['Accuracy'].apply(
        lambda x: '#2ecc71' if x >= 0.70 else '#f39c12' if x >= 0.60 else '#e74c3c'
    )

    fig.add_trace(go.Bar(
        y=team_df['Team'],
        x=team_df['Accuracy'],
        orientation='h',
        marker=dict(color=colors),
        text=team_df['Accuracy'].apply(lambda x: f'{x:.1%}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                      'Accuracy: %{x:.1%}<br>' +
                      'Games: %{customdata[0]}<br>' +
                      'Correct: %{customdata[1]}<br>' +
                      '<extra></extra>',
        customdata=team_df[['Games', 'Correct']].values
    ))

    fig.update_layout(
        title="Model Accuracy by Team (Ranked)",
        xaxis_title="Accuracy",
        yaxis_title="",
        xaxis=dict(tickformat='.0%', range=[0, 1.05]),
        height=max(400, len(team_df) * 25),  # Dynamic height based on number of teams
        showlegend=False,
        hovermode='y unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        best_team = team_df.iloc[-1]
        st.metric("Best Team", best_team['Team'], f"{best_team['Accuracy']:.1%}")
    with col2:
        worst_team = team_df.iloc[0]
        st.metric("Worst Team", worst_team['Team'], f"{worst_team['Accuracy']:.1%}")
    with col3:
        avg_accuracy = team_df['Accuracy'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.1%}")


def _render_recent_games_table(frame: pd.DataFrame) -> None:
    """Render recent games table with conditional formatting for errors."""
    st.subheader("📋 Recent Games Detail")

    with st.expander("ℹ️ What does this table show?", expanded=False):
        st.markdown(
            """
            This table shows **detailed game-by-game results** with error highlighting.

            - **Red highlighting**: Model was incorrect
            - **Darker red**: Incorrect prediction with 60-70% probability (concerning)
            - **Darkest red**: Incorrect prediction with 70%+ probability (very concerning)
            - **Green**: Model was correct

            This helps identify where the model is overconfident or making surprising mistakes.
            """
        )

    if frame.empty:
        st.info("No recent games available.")
        return

    # Prepare display dataframe
    display_df = frame.sort_values('game_date', ascending=False).copy()

    # Determine which teams played and who won
    display_df['Matchup'] = display_df.apply(
        lambda row: f"{row['away_team']} @ {row['home_team']}"
        if pd.notna(row.get('away_team')) and pd.notna(row.get('home_team'))
        else "Unknown",
        axis=1
    )

    # Convert predicted_winner from home/away to team abbreviation
    display_df['Predicted Winner'] = display_df.apply(
        lambda row: row['home_team'] if row.get('predicted_winner') == 'home'
        else row['away_team'] if row.get('predicted_winner') == 'away'
        else row.get('predicted_winner', 'N/A'),
        axis=1
    )

    # Convert actual_winner from home/away to team abbreviation
    display_df['Actual Winner'] = display_df.apply(
        lambda row: row['home_team'] if row.get('actual_winner') == 'home'
        else row['away_team'] if row.get('actual_winner') == 'away'
        else row.get('actual_winner', 'N/A'),
        axis=1
    )

    # Add score if available
    if 'actual_home_score' in display_df.columns and 'actual_away_score' in display_df.columns:
        display_df['Score'] = display_df.apply(
            lambda row: f"{int(row['actual_away_score'])}-{int(row['actual_home_score'])}"
            if pd.notna(row['actual_home_score']) and pd.notna(row['actual_away_score'])
            else "N/A",
            axis=1
        )
    else:
        display_df['Score'] = "N/A"

    # Format probability (renamed from confidence)
    display_df['Probability'] = display_df['predicted_probability'].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
    )

    # Create error severity column for styling
    display_df['Error_Severity'] = display_df.apply(
        lambda row: (
            'none' if row['correct']
            else 'high' if row['predicted_probability'] >= 0.70
            else 'medium' if row['predicted_probability'] >= 0.60
            else 'low'
        ),
        axis=1
    )

    # Select and rename columns for display
    table_df = display_df[[
        'game_date', 'Matchup', 'Predicted Winner', 'Probability',
        'Actual Winner', 'Score', 'correct', 'Error_Severity'
    ]].copy()

    table_df.columns = [
        'Date', 'Matchup', 'Predicted Winner', 'Probability',
        'Actual Winner', 'Score', 'Correct', 'Error_Severity'
    ]

    # Format date
    table_df['Date'] = pd.to_datetime(table_df['Date']).dt.strftime('%m/%d/%Y')

    # Format correct column
    table_df['Correct'] = table_df['Correct'].apply(lambda x: '✓' if x else '✗')

    # Add filters with color-coded labels
    st.markdown("**Filter Options:**")

    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])

    with col1:
        st.markdown('<span style="color: #2ecc71; font-weight: 600;">✓ Correct Predictions</span>', unsafe_allow_html=True)
        show_correct = st.checkbox("", value=True, key="show_correct", label_visibility="collapsed")
    with col2:
        st.markdown('<span style="color: #e74c3c; font-weight: 600;">High Prob Errors (70%+)</span>', unsafe_allow_html=True)
        show_high = st.checkbox("", value=True, key="show_high", label_visibility="collapsed")
    with col3:
        st.markdown('<span style="color: #f39c12; font-weight: 600;">Medium Prob Errors (60-70%)</span>', unsafe_allow_html=True)
        show_medium = st.checkbox("", value=True, key="show_medium", label_visibility="collapsed")
    with col4:
        st.markdown('<span style="color: #e08283; font-weight: 600;">Low Prob Errors (&lt;60%)</span>', unsafe_allow_html=True)
        show_low = st.checkbox("", value=True, key="show_low", label_visibility="collapsed")
    with col5:
        st.markdown("**Max Games**")
        max_games = st.number_input("", min_value=10, max_value=100, value=25, step=5, key="max_games", label_visibility="collapsed")

    # Build severity filter list based on checkboxes
    severity_filter = []
    if show_correct:
        severity_filter.append('none')
    if show_high:
        severity_filter.append('high')
    if show_medium:
        severity_filter.append('medium')
    if show_low:
        severity_filter.append('low')

    # Filter table based on selection
    if not severity_filter:
        st.warning("Please select at least one filter option.")
        return

    filtered_table = table_df[table_df['Error_Severity'].isin(severity_filter)]

    # Limit to most recent games (configurable)
    filtered_table = filtered_table.head(max_games)

    # Check if any games match the filter
    if filtered_table.empty:
        st.info("No games match the selected filters.")
        return

    # Apply styling - determine color based on Error_Severity column
    def highlight_errors(row):
        # Get the error severity from the filtered table
        severity = filtered_table.loc[row.name, 'Error_Severity']

        # Return colors for display_table columns (7 columns after dropping Error_Severity)
        # Columns: Date, Matchup, Predicted Winner, Probability, Actual Winner, Score, Correct
        if severity == 'none':
            return ['background-color: #d5f4e6'] * 7  # Light green
        elif severity == 'high':
            return ['background-color: #e74c3c; color: white'] * 7  # Dark red
        elif severity == 'medium':
            return ['background-color: #f39c12; color: white'] * 7  # Orange
        else:  # low
            return ['background-color: #fadbd8'] * 7  # Light red

    # Drop the Error_Severity column before displaying
    display_table = filtered_table.drop('Error_Severity', axis=1)

    # Apply styling and display
    styled_df = display_table.style.apply(highlight_errors, axis=1)

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Summary statistics for visible games
    num_errors = len(filtered_table[filtered_table['Correct'] == '✗'])
    high_prob_errors = len(filtered_table[filtered_table['Error_Severity'] == 'high'])
    medium_prob_errors = len(filtered_table[filtered_table['Error_Severity'] == 'medium'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Games Shown", len(filtered_table))
    with col2:
        st.metric("Total Errors", num_errors)
    with col3:
        st.metric("High Probability Errors (70%+)", high_prob_errors)
    with col4:
        st.metric("Medium Probability Errors (60-70%)", medium_prob_errors)


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
        "predicted_winner_prob",      # Normalized probability (predicted winner's prob)
        "calibrated_home_win_prob",   # Raw home team probability
        "home_win_probability",
        "home_win_prob",
    ]:
        if column in dataset.columns:
            return column
    return None


def _derive_actual_outcome(dataset: pd.DataFrame) -> pd.Series:
    # If we have the pre-calculated predicted_winner_won column, use it
    # This is 1 if the predicted winner actually won, 0 otherwise
    if "predicted_winner_won" in dataset.columns:
        return dataset["predicted_winner_won"].astype(float)

    # Fallback: Calculate from actual scores (for home team)
    if {"actual_home_score", "actual_away_score"}.issubset(dataset.columns):
        return (dataset["actual_home_score"] > dataset["actual_away_score"]).astype(float)

    # Fallback: Calculate from actual winner (for home team)
    if {"actual_winner", "home_team"}.issubset(dataset.columns):
        return (dataset["actual_winner"] == dataset["home_team"]).astype(float)

    # Fallback: Use prediction correctness
    if "prediction_correct" in dataset.columns:
        return dataset["prediction_correct"].astype(float)

    return pd.Series([np.nan] * len(dataset), index=dataset.index)


if __name__ == "__main__":
    main()
