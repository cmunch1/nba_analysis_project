"""
Calibration chart components for Historical Performance page.

Provides reusable functions for creating:
- Sharpness + Calibration by probability bin charts
- Threshold error rate charts
- Brier score calculations
"""

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def prepare_calibration_bins(
    df: pd.DataFrame,
    probability_col: str = "predicted_probability",
    outcome_col: str = "actual_outcome",
    n_bins: int = 20,
    min_prob: float = 0.5,
) -> pd.DataFrame:
    """
    Bin predictions and calculate actual win rates for calibration analysis.

    Args:
        df: DataFrame with predictions and outcomes
        probability_col: Column name for predicted probabilities
        outcome_col: Column name for actual outcomes (0 or 1)
        n_bins: Number of probability bins (default 20 = 2.5% increments)
        min_prob: Minimum probability to include (default 0.5)

    Returns:
        DataFrame with columns:
        - bin_label: String label for bin (e.g., "0.50-0.525")
        - bin_center: Midpoint of bin
        - avg_predicted: Average predicted probability in bin
        - actual_rate: Actual win rate in bin
        - count: Number of predictions in bin
        - brier_score: Brier score for bin
    """
    # Filter to minimum probability
    df_filtered = df[df[probability_col] >= min_prob].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # Create bins from min_prob to 1.0
    bin_edges = np.linspace(min_prob, 1.0, n_bins + 1)
    df_filtered["bin"] = pd.cut(
        df_filtered[probability_col],
        bins=bin_edges,
        include_lowest=True,
    )

    # Calculate statistics per bin
    bin_stats = (
        df_filtered.groupby("bin", observed=False)
        .agg(
            avg_predicted=(probability_col, "mean"),
            actual_rate=(outcome_col, "mean"),
            count=(outcome_col, "size"),
        )
        .reset_index()
    )

    # Remove empty bins
    bin_stats = bin_stats[bin_stats["count"] > 0].copy()

    if bin_stats.empty:
        return pd.DataFrame()

    # Calculate Brier score for each bin
    merged = df_filtered.merge(
        bin_stats[["bin", "avg_predicted"]],
        on="bin",
        how="left",
    )
    brier_by_bin = (
        merged.groupby("bin", observed=False)
        .apply(
            lambda x: np.mean((x[probability_col] - x[outcome_col]) ** 2),
            include_groups=False,
        )
        .reset_index(name="brier_score")
    )
    bin_stats = bin_stats.merge(brier_by_bin, on="bin", how="left")

    # Create readable labels
    def format_bin_label(interval) -> str:
        return f"{interval.left:.3f}-{interval.right:.3f}"

    bin_stats["bin_label"] = bin_stats["bin"].apply(format_bin_label)
    bin_stats["bin_center"] = bin_stats["bin"].apply(lambda x: (x.left + x.right) / 2)

    return bin_stats[
        [
            "bin_label",
            "bin_center",
            "avg_predicted",
            "actual_rate",
            "count",
            "brier_score",
        ]
    ]


def calculate_threshold_metrics(
    df: pd.DataFrame,
    probability_col: str = "predicted_probability",
    outcome_col: str = "actual_outcome",
    correct_col: str = "correct",
    min_threshold: float = 0.5,
    max_threshold: float = 1.0,
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Calculate error rate and accuracy at each probability threshold.

    Args:
        df: DataFrame with predictions and outcomes
        probability_col: Column name for predicted probabilities
        outcome_col: Column name for actual outcomes
        correct_col: Column name for correctness indicator
        min_threshold: Minimum threshold to evaluate
        max_threshold: Maximum threshold to evaluate
        step: Step size between thresholds

    Returns:
        DataFrame with columns:
        - threshold: Probability threshold
        - count: Number of predictions >= threshold
        - accuracy: Accuracy of predictions >= threshold
        - error_rate: Error rate (1 - accuracy)
    """
    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)
    thresholds = np.round(thresholds, 3)

    rows = []
    for threshold in thresholds:
        subset = df[df[probability_col] >= threshold]
        n = len(subset)

        if n > 0:
            accuracy = subset[correct_col].mean()
            error_rate = 1 - accuracy
        else:
            accuracy = np.nan
            error_rate = np.nan

        rows.append(
            {
                "threshold": threshold,
                "count": n,
                "accuracy": accuracy,
                "error_rate": error_rate,
            }
        )

    return pd.DataFrame(rows)


def calculate_brier_score(
    df: pd.DataFrame,
    probability_col: str = "predicted_probability",
    outcome_col: str = "actual_outcome",
) -> float:
    """
    Calculate overall Brier score for predictions.

    Args:
        df: DataFrame with predictions and outcomes
        probability_col: Column name for predicted probabilities
        outcome_col: Column name for actual outcomes (0 or 1)

    Returns:
        Brier score (lower is better, range 0-1)
    """
    if df.empty:
        return np.nan

    predictions = df[probability_col].values
    outcomes = df[outcome_col].values

    return float(np.mean((predictions - outcomes) ** 2))


def create_calibration_bin_chart(bin_stats: pd.DataFrame) -> go.Figure:
    """
    Create combined sharpness + calibration chart.

    Shows:
    - Bar chart: count of predictions per bin (secondary y-axis)
    - Line + scatter: actual win rate per bin (primary y-axis)
    - Diagonal reference line (perfect calibration)

    Args:
        bin_stats: DataFrame from prepare_calibration_bins()

    Returns:
        Plotly figure
    """
    if bin_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for calibration binning",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for counts (secondary y-axis)
    fig.add_trace(
        go.Bar(
            x=bin_stats["bin_center"],
            y=bin_stats["count"],
            name="Prediction Count",
            marker_color="lightblue",
            opacity=0.6,
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + "Count: %{y}<br>"
            + "Avg Predicted: %{customdata[1]:.1%}<br>"
            + "Actual Rate: %{customdata[2]:.1%}<br>"
            + "Brier Score: %{customdata[3]:.4f}<br>"
            + "<extra></extra>",
            customdata=np.column_stack(
                (
                    bin_stats["bin_label"],
                    bin_stats["avg_predicted"],
                    bin_stats["actual_rate"],
                    bin_stats["brier_score"],
                )
            ),
        ),
        secondary_y=True,
    )

    # Line + scatter for actual win rate (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=bin_stats["bin_center"],
            y=bin_stats["actual_rate"],
            mode="lines+markers",
            name="Actual Win Rate",
            line=dict(color="darkblue", width=2),
            marker=dict(size=8, color="darkblue"),
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + "Actual Rate: %{y:.1%}<br>"
            + "Avg Predicted: %{customdata[1]:.1%}<br>"
            + "Count: %{customdata[2]}<br>"
            + "<extra></extra>",
            customdata=np.column_stack(
                (
                    bin_stats["bin_label"],
                    bin_stats["avg_predicted"],
                    bin_stats["count"],
                )
            ),
        ),
        secondary_y=False,
    )

    # Perfect calibration line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[0.5, 1.0],
            y=[0.5, 1.0],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash", width=2),
            showlegend=True,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    # Update axes
    fig.update_xaxes(
        title_text="Predicted Win Probability",
        tickformat=".0%",
        range=[0.48, 1.02],
    )
    fig.update_yaxes(
        title_text="Actual Win Rate",
        tickformat=".0%",
        range=[0, 1.05],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Number of Predictions",
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title="Model Sharpness & Calibration by Probability Bin",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=500,
    )

    return fig


def create_threshold_error_chart(threshold_stats: pd.DataFrame) -> go.Figure:
    """
    Create error rate vs confidence threshold chart.

    Shows:
    - Line chart: error rate at each threshold (primary y-axis)
    - Area chart: sample size at each threshold (secondary y-axis)
    - Reference lines for common thresholds (60%, 70%, 80%)

    Args:
        threshold_stats: DataFrame from calculate_threshold_metrics()

    Returns:
        Plotly figure
    """
    if threshold_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for threshold analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Area chart for sample size (secondary y-axis, behind)
    fig.add_trace(
        go.Scatter(
            x=threshold_stats["threshold"],
            y=threshold_stats["count"],
            mode="lines",
            fill="tozeroy",
            name="Sample Size",
            line=dict(color="lightgray", width=0),
            fillcolor="rgba(211, 211, 211, 0.3)",
            hovertemplate="Threshold: %{x:.0%}<br>"
            + "Games ≥ threshold: %{y}<br>"
            + "<extra></extra>",
        ),
        secondary_y=True,
    )

    # Line chart for error rate (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=threshold_stats["threshold"],
            y=threshold_stats["error_rate"],
            mode="lines",
            name="Error Rate",
            line=dict(color="red", width=3),
            hovertemplate="Threshold: %{x:.0%}<br>"
            + "Error Rate: %{y:.1%}<br>"
            + "Accuracy: %{customdata[0]:.1%}<br>"
            + "Games: %{customdata[1]}<br>"
            + "<extra></extra>",
            customdata=np.column_stack(
                (threshold_stats["accuracy"], threshold_stats["count"])
            ),
        ),
        secondary_y=False,
    )

    # Add reference lines for common thresholds
    for threshold_value, color in [(0.6, "green"), (0.7, "orange"), (0.8, "purple")]:
        # Find stats at this threshold
        matching = threshold_stats[
            np.isclose(threshold_stats["threshold"], threshold_value, atol=0.005)
        ]
        if not matching.empty:
            row = matching.iloc[0]
            fig.add_vline(
                x=threshold_value,
                line_dash="dot",
                line_color=color,
                opacity=0.5,
                annotation_text=f"{threshold_value:.0%}",
                annotation_position="top",
            )

    # Update axes
    fig.update_xaxes(
        title_text="Minimum Predicted Win Probability",
        tickformat=".0%",
        range=[0.48, 1.02],
    )
    fig.update_yaxes(
        title_text="Error Rate (for predictions ≥ threshold)",
        tickformat=".0%",
        range=[0, max(0.5, threshold_stats["error_rate"].max() * 1.1)],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Number of Games",
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title="Error Rate vs Confidence Threshold",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=500,
    )

    return fig


def create_threshold_summary_table(threshold_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary table for key thresholds.

    Args:
        threshold_stats: DataFrame from calculate_threshold_metrics()

    Returns:
        DataFrame with key threshold metrics formatted for display
    """
    if threshold_stats.empty:
        return pd.DataFrame()

    # Select key thresholds
    key_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []

    for threshold_value in key_thresholds:
        matching = threshold_stats[
            np.isclose(threshold_stats["threshold"], threshold_value, atol=0.005)
        ]
        if not matching.empty:
            row = matching.iloc[0]
            rows.append(
                {
                    "Threshold": f"{threshold_value:.0%}",
                    "Games": int(row["count"]),
                    "Accuracy": f"{row['accuracy']:.1%}" if not pd.isna(row["accuracy"]) else "N/A",
                    "Error Rate": f"{row['error_rate']:.1%}" if not pd.isna(row["error_rate"]) else "N/A",
                }
            )

    return pd.DataFrame(rows)
