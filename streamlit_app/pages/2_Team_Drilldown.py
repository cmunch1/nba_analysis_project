"""Team Drilldown Page

Displays team-specific performance metrics using pre-calculated data from the dashboard prep pipeline.
All calculations are performed in dashboard prep - this page only visualizes the data.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    """Main function for Team Drilldown page."""
    st.title("Team Drilldown")
    st.caption("Deep-dive into model performance for a specific team using pre-calculated metrics from the dashboard prep pipeline.")

    dataset = load_latest_dataset()
    if dataset.empty:
        st.info("No dashboard data available yet.")
        return

    # Extract team list from team_performance section
    team_performance = dataset[dataset['section'] == 'team_performance'].copy()
    results = dataset[dataset['section'] == 'results'].copy()

    if team_performance.empty:
        st.warning("Team performance data not yet available. Ensure the dashboard prep pipeline has run with sufficient validated results.")
        return

    teams = sorted(team_performance['team_name'].dropna().unique().tolist())
    if not teams:
        st.warning("No teams found in performance data.")
        return

    # Sidebar: Team selection
    with st.sidebar:
        st.subheader("Team Selection")
        selected_team = st.selectbox("Team", teams, key="team_selector")
        max_rows = st.slider("Recent games to display", min_value=5, max_value=50, value=15, step=5)

    # Get team metrics from pre-calculated data
    team_metrics = team_performance[team_performance['team_name'] == selected_team]
    if team_metrics.empty:
        st.info(f"No performance data available for {selected_team} yet.")
        return

    # Get team-specific game results
    team_results = _filter_team_results(results, selected_team)

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
        st.markdown('<div class="nav-button">📊 <a href="#team-summary">Team Summary</a></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="nav-button">🏠 <a href="#home-away">Home vs Away</a></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="nav-button">📈 <a href="#performance-trend">Performance Trend</a></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="nav-button">📋 <a href="#recent-games">Recent Games</a></div>', unsafe_allow_html=True)
    st.markdown("---")

    # Render sections with anchor points
    st.markdown('<div id="team-summary"></div>', unsafe_allow_html=True)
    _render_team_summary(team_metrics.iloc[0])

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="home-away"></div>', unsafe_allow_html=True)
    _render_home_away_performance(team_metrics.iloc[0], team_results)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="performance-trend"></div>', unsafe_allow_html=True)
    _render_recent_performance_trend(team_results)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="recent-games"></div>', unsafe_allow_html=True)
    _render_recent_games_table(team_results, max_rows)


def _filter_team_results(results: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Filter results to games involving the selected team.

    Args:
        results: All results data
        team: Team name/abbreviation

    Returns:
        DataFrame with team-specific results sorted by date descending
    """
    if results.empty:
        return pd.DataFrame()

    # Filter to games where team was involved
    team_mask = (
        (results.get('home_team') == team) |
        (results.get('away_team') == team)
    )
    team_results = results[team_mask].copy()

    if team_results.empty:
        return team_results

    # Add helper columns
    team_results['game_date'] = pd.to_datetime(team_results.get('game_date'), errors='coerce')
    team_results['is_home'] = team_results.get('home_team') == team
    team_results['opponent'] = np.where(
        team_results['is_home'],
        team_results.get('away_team'),
        team_results.get('home_team')
    )

    # Determine if team won
    team_results['team_won'] = (
        (team_results['is_home'] & (team_results.get('actual_winner') == 'home')) |
        (~team_results['is_home'] & (team_results.get('actual_winner') == 'away'))
    )

    # Determine if prediction was correct
    team_results['correct_prediction'] = team_results.get('prediction_correct', np.nan)

    # Sort by date descending
    team_results = team_results.sort_values('game_date', ascending=False)

    return team_results


def _render_team_summary(team_row: pd.Series) -> None:
    """
    Display overall team performance metrics from pre-calculated data.

    Args:
        team_row: Row from team_performance section with pre-calculated metrics
    """
    st.subheader("Team Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    total_predictions = int(team_row.get('total_predictions', 0))
    overall_accuracy = pd.to_numeric(team_row.get('accuracy'), errors='coerce')
    avg_prob = pd.to_numeric(team_row.get('avg_predicted_probability'), errors='coerce')

    col1.metric("Total Games Predicted", total_predictions)

    if pd.notna(overall_accuracy):
        col2.metric("Overall Accuracy", f"{overall_accuracy * 100:.1f}%")
    else:
        col2.metric("Overall Accuracy", "N/A")

    if pd.notna(avg_prob):
        col3.metric("Avg Win Probability", f"{avg_prob * 100:.1f}%")
    else:
        col3.metric("Avg Win Probability", "N/A")

    # Calculate home game percentage
    home_games = team_row.get('home_games', 0)
    if total_predictions > 0:
        home_pct = (home_games / total_predictions) * 100
        col4.metric("Home Games", f"{int(home_games)} ({home_pct:.0f}%)")
    else:
        col4.metric("Home Games", "N/A")

    st.caption("All metrics calculated by the dashboard prep pipeline from validated predictions and actual results.")


def _render_home_away_performance(team_row: pd.Series, team_results: pd.DataFrame) -> None:
    """
    Display home vs away prediction accuracy AND actual win rate.

    Args:
        team_row: Row from team_performance section with pre-calculated metrics
        team_results: Team-specific results data for calculating actual win rates
    """
    st.subheader("Home vs Away Performance")

    home_accuracy = pd.to_numeric(team_row.get('home_accuracy'), errors='coerce')
    away_accuracy = pd.to_numeric(team_row.get('away_accuracy'), errors='coerce')
    home_games = int(team_row.get('home_games', 0))
    away_games = int(team_row.get('away_games', 0))

    # Calculate actual win rates from results data
    home_win_rate = None
    away_win_rate = None

    if not team_results.empty and 'is_home' in team_results.columns and 'team_won' in team_results.columns:
        home_results = team_results[team_results['is_home'] == True]
        away_results = team_results[team_results['is_home'] == False]

        if len(home_results) > 0:
            home_win_rate = home_results['team_won'].mean()

        if len(away_results) > 0:
            away_win_rate = away_results['team_won'].mean()

    # Build data for grouped bar chart
    chart_data = []

    # Home games data
    if pd.notna(home_accuracy) and home_games > 0:
        chart_data.append({
            'Location': 'Home',
            'Metric': 'Prediction Accuracy',
            'Value': home_accuracy,
            'Games': home_games
        })

    if home_win_rate is not None:
        chart_data.append({
            'Location': 'Home',
            'Metric': 'Actual Win Rate',
            'Value': home_win_rate,
            'Games': home_games
        })

    # Away games data
    if pd.notna(away_accuracy) and away_games > 0:
        chart_data.append({
            'Location': 'Away',
            'Metric': 'Prediction Accuracy',
            'Value': away_accuracy,
            'Games': away_games
        })

    if away_win_rate is not None:
        chart_data.append({
            'Location': 'Away',
            'Metric': 'Actual Win Rate',
            'Value': away_win_rate,
            'Games': away_games
        })

    if not chart_data:
        st.info("Insufficient data to display home/away performance breakdown.")
        return

    chart_df = pd.DataFrame(chart_data)

    # Create grouped bar chart
    fig = px.bar(
        chart_df,
        x='Location',
        y='Value',
        color='Metric',
        barmode='group',
        text='Value',
        hover_data={'Games': True, 'Value': ':.1%'},
        labels={'Value': 'Rate'},
        color_discrete_map={
            'Prediction Accuracy': '#636EFA',  # Blue
            'Actual Win Rate': '#00CC96'      # Green
        }
    )

    fig.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside'
    )
    fig.update_yaxes(range=[0, 1.0], tickformat='.0%')

    # Update legend position
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display game counts and metrics summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Home Games", home_games)
    col2.metric("Home Win Rate", f"{home_win_rate * 100:.1f}%" if home_win_rate is not None else "N/A")
    col3.metric("Away Games", away_games)
    col4.metric("Away Win Rate", f"{away_win_rate * 100:.1f}%" if away_win_rate is not None else "N/A")


def _render_recent_performance_trend(team_results: pd.DataFrame) -> None:
    """
    Display recent performance trends for the team.

    Args:
        team_results: Team-specific results data
    """
    st.subheader("Recent Performance Trend")

    if team_results.empty or 'game_date' not in team_results.columns:
        st.info("No recent results available to display trend.")
        return

    # Take most recent games (sorted descending, so reverse for chronological chart)
    recent = team_results.head(20).sort_values('game_date', ascending=True)

    if recent.empty:
        st.info("No recent results available.")
        return

    # Create figure with dual y-axis
    fig = go.Figure()

    # Add win/loss indicator as markers/bars
    if 'team_won' in recent.columns:
        # Show wins as green markers at y=1
        wins = recent[recent['team_won'] == True]
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins['game_date'],
                y=[1] * len(wins),
                name='Win',
                mode='markers',
                marker=dict(color='green', size=15, symbol='square'),
                yaxis='y2',
                hovertemplate='%{x}<br>Result: Win<extra></extra>'
            ))

        # Show losses as red markers at y=0
        losses = recent[recent['team_won'] == False]
        if not losses.empty:
            fig.add_trace(go.Scatter(
                x=losses['game_date'],
                y=[0] * len(losses),
                name='Loss',
                mode='markers',
                marker=dict(color='red', size=15, symbol='square'),
                yaxis='y2',
                hovertemplate='%{x}<br>Result: Loss<extra></extra>'
            ))

    # Add prediction correctness as line
    if 'correct_prediction' in recent.columns:
        fig.add_trace(go.Scatter(
            x=recent['game_date'],
            y=recent['correct_prediction'].astype(float),
            name='Correct Prediction',
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            yaxis='y2'
        ))

    # Add win probability if available
    if 'predicted_winner_prob' in recent.columns:
        # Normalize to team's perspective
        recent['team_win_prob'] = recent.apply(
            lambda row: row['predicted_winner_prob']
            if (row.get('predicted_winner') == 'home' and row.get('is_home')) or
               (row.get('predicted_winner') == 'away' and not row.get('is_home'))
            else (1 - row['predicted_winner_prob']) if pd.notna(row.get('predicted_winner_prob')) else np.nan,
            axis=1
        )

        fig.add_trace(go.Scatter(
            x=recent['game_date'],
            y=recent['team_win_prob'],
            name='Predicted Win Probability',
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            yaxis='y'
        ))

    # Update layout with dual y-axes
    fig.update_layout(
        xaxis_title='Game Date',
        yaxis=dict(
            title=dict(text='Win Probability', font=dict(color='orange')),
            tickformat='.0%',
            range=[0, 1]
        ),
        yaxis2=dict(
            title=dict(text='Result (1=Win, 0=Loss)', font=dict(color='green')),
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Green squares = wins, Red squares = losses, Blue line = correct predictions, Orange dashed = predicted win probability")


def _render_recent_games_table(team_results: pd.DataFrame, max_rows: int) -> None:
    """
    Display recent games table with conditional formatting matching Historical Performance page.

    Args:
        team_results: Team-specific results data
        max_rows: Maximum number of rows to display
    """
    st.subheader("📋 Recent Games Detail")

    with st.expander("ℹ️ What does this table show?", expanded=False):
        st.markdown(
            """
            This table shows **detailed game-by-game results** for this team with error highlighting.

            - **Red highlighting**: Model was incorrect
            - **Darker red**: Incorrect prediction with 60-70% probability (concerning)
            - **Darkest red**: Incorrect prediction with 70%+ probability (very concerning)
            - **Green**: Model was correct

            This helps identify where the model is overconfident or making surprising mistakes for this specific team.
            """
        )

    if team_results.empty:
        st.info("No recent games available to display.")
        return

    # Prepare display dataframe
    display_df = team_results.copy()

    # Create matchup string (opponent at location)
    display_df['Matchup'] = display_df.apply(
        lambda row: f"vs {row['opponent']}" if row.get('is_home')
        else f"@ {row['opponent']}" if pd.notna(row.get('opponent'))
        else "Unknown",
        axis=1
    )

    # Add score if available
    if 'actual_home_score' in display_df.columns and 'actual_away_score' in display_df.columns:
        display_df['Score'] = display_df.apply(
            lambda row: (
                f"{int(row['actual_away_score'])}-{int(row['actual_home_score'])}"
                if row.get('is_home')
                else f"{int(row['actual_home_score'])}-{int(row['actual_away_score'])}"
            ) if pd.notna(row.get('actual_home_score')) and pd.notna(row.get('actual_away_score'))
            else "N/A",
            axis=1
        )
    else:
        display_df['Score'] = "N/A"

    # Format probability
    probability_col = 'predicted_winner_prob'
    if probability_col in display_df.columns:
        display_df['Probability'] = display_df[probability_col].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        # Keep numeric version for error severity calculation
        display_df['prob_numeric'] = pd.to_numeric(display_df[probability_col], errors='coerce')
    else:
        display_df['Probability'] = "N/A"
        display_df['prob_numeric'] = 0.5

    # Determine result
    display_df['Result'] = display_df['team_won'].apply(lambda x: 'Win' if x else 'Loss')

    # Create error severity column for styling
    display_df['correct'] = display_df.get('correct_prediction', False)
    display_df['Error_Severity'] = display_df.apply(
        lambda row: (
            'none' if row.get('correct')
            else 'high' if row.get('prob_numeric', 0) >= 0.70
            else 'medium' if row.get('prob_numeric', 0) >= 0.60
            else 'low'
        ),
        axis=1
    )

    # Format correct column
    display_df['Correct'] = display_df['correct'].apply(lambda x: '✓' if x else '✗')

    # Select columns for display
    table_df = display_df[[
        'game_date', 'Matchup', 'Result', 'Probability',
        'Score', 'Correct', 'Error_Severity'
    ]].copy()

    # Rename columns
    table_df.columns = [
        'Date', 'Matchup', 'Result', 'Probability',
        'Score', 'Correct', 'Error_Severity'
    ]

    # Format date
    table_df['Date'] = pd.to_datetime(table_df['Date']).dt.strftime('%m/%d/%Y')

    # Add filters with color-coded labels
    st.markdown("**Filter Options:**")

    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])

    with col1:
        st.markdown('<span style="color: #2ecc71; font-weight: 600;">✓ Correct Predictions</span>', unsafe_allow_html=True)
        show_correct = st.checkbox("", value=True, key="team_show_correct", label_visibility="collapsed")
    with col2:
        st.markdown('<span style="color: #e74c3c; font-weight: 600;">High Prob Errors (70%+)</span>', unsafe_allow_html=True)
        show_high = st.checkbox("", value=True, key="team_show_high", label_visibility="collapsed")
    with col3:
        st.markdown('<span style="color: #f39c12; font-weight: 600;">Medium Prob Errors (60-70%)</span>', unsafe_allow_html=True)
        show_medium = st.checkbox("", value=True, key="team_show_medium", label_visibility="collapsed")
    with col4:
        st.markdown('<span style="color: #e08283; font-weight: 600;">Low Prob Errors (&lt;60%)</span>', unsafe_allow_html=True)
        show_low = st.checkbox("", value=True, key="team_show_low", label_visibility="collapsed")
    with col5:
        st.markdown("**Max Games**")
        max_games = st.number_input("", min_value=5, max_value=50, value=max_rows, step=5, key="team_max_games", label_visibility="collapsed")

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

    # Limit to most recent games
    filtered_table = filtered_table.head(max_games)

    # Check if any games match the filter
    if filtered_table.empty:
        st.info("No games match the selected filters.")
        return

    # Apply styling - determine color based on Error_Severity column
    def highlight_errors(row):
        severity = filtered_table.loc[row.name, 'Error_Severity']

        # Return colors for display_table columns (6 columns after dropping Error_Severity)
        # Columns: Date, Matchup, Result, Probability, Score, Correct
        if severity == 'none':
            return ['background-color: #d5f4e6'] * 6  # Light green
        elif severity == 'high':
            return ['background-color: #e74c3c; color: white'] * 6  # Dark red
        elif severity == 'medium':
            return ['background-color: #f39c12; color: white'] * 6  # Orange
        else:  # low
            return ['background-color: #fadbd8'] * 6  # Light red

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


if __name__ == "__main__":
    main()
