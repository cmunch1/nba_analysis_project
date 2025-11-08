# Streamlit Dashboard Reference Guide

This guide summarizes the Streamlit implementation that powers the NBA predictions control panel. Use it when running the app locally, extending UI components, or wiring data from the nightly pipeline.

## Project Layout

```
streamlit_app/
├── app.py                      # Auto-redirect landing page
├── data_loader.py              # Cached data access helpers (DI-backed)
├── di_container.py             # Streamlit-specific dependency container
├── README.md                   # Quick start instructions
├── components/
│   ├── __init__.py
│   ├── calibration_charts.py   # Calibration analysis and threshold metrics
│   ├── kpi_panel.py            # Summary metric display
│   └── matchup_cards.py        # Filter controls + matchup grid rendering
└── pages/
    ├── 0_Todays_Predictions.py # Main predictions view
    ├── 1_Historical_Performance.py # Historical results and accuracy analysis
    └── 2_Model_Analysis.py     # Technical model diagnostics
```

All components pull configuration, logging, and file handling from `ml_framework.core.common_di_container.CommonDIContainer` to stay aligned with the rest of the project.

## Running the App

1. Ensure the nightly pipeline has produced `data/dashboard/dashboard_data.csv` (and optional archives in `data/dashboard/archive/`).
2. Install dependencies (only needed after changes to `pyproject.toml`):
   ```bash
   uv sync
   ```
3. Launch Streamlit from the project root:
   ```bash
   uv run streamlit run streamlit_app/app.py
   ```
4. Open the URL Streamlit prints (default http://localhost:8501).

The landing page automatically loads the latest dashboard dataset and allows switching to archived snapshots through the sidebar.

## Landing Page (`app.py`)

- **Header KPI Metrics:** Displays 5 key performance indicators extracted from the dashboard_data.csv metrics section:
  - **Season Accuracy:** Overall model accuracy across all validated games
  - **7-Day Accuracy:** Recent performance window for model drift detection
  - **Total Games Predicted:** Count of all predictions with validated results
  - **Total Predictions Today:** Count of today's matchup predictions
  - **High Probability Games:** Games above the configured threshold (default 60%)
- **Filter panel (sidebar):** Drives matchup filtering by date, team, and probability threshold. Default threshold uses the configured `dashboard_prep.predictions.high_probability_threshold`.
- **Matchup grid:** Card-based layout with the following improvements:
  - **Full Team Names:** Displays "Houston Rockets @ Boston Celtics" instead of "HOU vs BOS"
  - **Matchup Format:** "Away @ Home" format (visitor team at home team's court)
  - **Predicted Winner:** Shows actual team name (e.g., "Boston Celtics") instead of generic "home"/"away"
  - **Win Probability:** Displays the predicted winner's calibrated probability (not always home team)
  - **Reliability Chips:** "High Win Probability" for games ≥60%, "Recent Accuracy" based on historical performance
- **Dataset preview:** Expandable table for quick inspection of the raw dashboard output in its current state.
- **Section Filtering:** Automatically filters to show only 'predictions' section (excludes 'results' and 'metrics')

Because the filters, KPIs, and cards operate on the same data frame, you only need to add new columns to the nightly CSV and reference them in the component renderers to surface new information.

## Components

- `matchup_cards.render_filter_panel`: Sidebar UI; update here when introducing new filters (e.g., head-to-head toggle).
- `matchup_cards.render_matchup_grid`: Controls card layout and reliability chips. Recent updates include:
  - **Team Name Mapping:** Loads full team names from `configs/nba/team_mapping.yaml` via DI container
  - **SimpleNamespace Handling:** Converts config object to dict using `vars()` for `.get()` access
  - **Win Probability Logic:** Calculates `1 - home_win_prob` when away team is predicted winner
  - **Font Size Styling:** Uses HTML with `font-size: 0.95em` for predicted team and probability values
  - **Terminology:** Uses `predicted_probability` column (calibrated probability of predicted winner)
  - Extend this function to display injuries, implied odds, or model-version metadata
- `kpi_panel.render_kpi_panel`: Summary metrics for the filtered dataset. Uses `predicted_probability` for threshold-based filtering. Add additional KPIs here (e.g., coverage rate) and ensure null-safe calculations.

All components tolerate missing columns by checking their presence before rendering metrics to keep the UI resilient while the pipeline evolves.

## Supporting Pages

Streamlit automatically exposes files under `streamlit_app/pages/` as additional tabs.

- **Historical Performance (1_Historical_Performance.py):**
  - **Data Source:** Uses pre-calculated metrics from dashboard prep pipeline (single source of truth)
  - **Section Filtering:** Automatically filters to show only 'results' section (games with actual outcomes)
  - **Individual Chart Filters:** Each section has its own time period filter (7/14/30/All days) instead of global filter
  - **Daily Performance:** Shows daily accuracy, season running accuracy, period average, and season average baseline with 7-day rolling window option
  - **Threshold Analysis:** Error rate vs predicted probability threshold chart with key threshold summary table
  - **Team Accuracy:** Horizontal bar chart ranking teams by model accuracy (minimum 5 games required)
  - **Recent Games Detail:** Game-by-game table with conditional formatting highlighting errors by probability level
  - **Navigation:** Jump links to each section for easy page navigation
  - **Required Columns:** `game_date`, `predicted_winner_prob` (or `predicted_probability`), `prediction_correct`, `predicted_winner_won`, `home_team`, `away_team`

- **Model Analysis (2_Model_Analysis.py):**
  - **Audience:** Technical users and ML practitioners
  - **Calibration Diagnostics:** 20-bin calibration curve with sharpness distribution, Brier score metrics, and bin-level statistics
  - **Drift Monitoring:** Accuracy over time with 7-day rolling average, Brier score trends, calibration error trends, and drift summary statistics
  - **Error Analysis:** High-level error metrics categorized by predicted probability level (placeholder for future detailed analysis)
  - **Individual Filters:** Each section has its own time period filter (14/30/All days for calibration, 14/30/All for drift)
  - **Data Source:** Uses pre-calculated drift metrics from dashboard prep pipeline
  - **Required Columns:** `predicted_winner_prob` (or `predicted_probability`), `predicted_winner_won`, `game_date`, drift section with `date`, `accuracy`, `brier_score`, `calibration_error`, `avg_predicted_probability`
  - **Terminology:** Uses "predicted probability" (calibrated) throughout, avoiding ambiguous terms like "confidence"

- **Team Drilldown (2_Team_Drilldown.py):**
  - **Purpose:** Deep-dive analysis of model performance for individual teams
  - **Data Source:** Uses pre-calculated team_performance section from dashboard prep pipeline (single source of truth)
  - **Team Performance Summary:** Overall accuracy, total games predicted, average win probability, home game percentage
  - **Home vs Away Performance:** Bar chart comparing prediction accuracy at home vs away (uses pre-calculated `home_accuracy` and `away_accuracy`)
  - **Recent Performance Trend:** Dual-axis chart showing actual results (win/loss bars), prediction correctness (blue line), and predicted win probability (orange dashed line) over last 20 games
  - **Recent Games Detail:** Configurable table (5-50 games) with date, location, opponent, result, prediction accuracy, win probability, and prediction error
  - **Required Sections:** `team_performance` section with `team_name`, `accuracy`, `home_accuracy`, `away_accuracy`, `home_games`, `away_games`, `total_predictions`, `avg_predicted_probability`
  - **Required Columns (results section):** `home_team`, `away_team`, `actual_winner`, `predicted_winner`, `predicted_winner_prob`, `prediction_correct`, `prediction_error`, `game_date`
  - **Design Principle:** All calculations performed in dashboard prep pipeline; UI only visualizes pre-calculated data
  - **Configuration:** Minimum games threshold controlled by `dashboard_prep.team_performance.min_games` (currently 5 for early season)

Add new analytical views by creating additional files in `streamlit_app/pages/`—re-use `data_loader.load_latest_dataset()` so each page shares the same cached data.

## Updating the Dashboard

1. **Add data columns:** Extend the dashboard prep pipeline so the nightly CSV includes the necessary fields. Keep metadata column names consistent (e.g., `home_team`, `away_team`).
2. **Update components:** Modify or create component functions to consume the new fields. Prefer adding helper methods inside `components/` for reuse.
3. **Adjust tests (future):** When smoke tests for the Streamlit utilities are introduced, update them alongside data schema changes.
4. **Document changes:** Update this guide and `streamlit_app/README.md` when feature additions require new setup instructions.

## Troubleshooting

- **No data available:** Confirm the nightly pipeline ran successfully and wrote `dashboard_data.csv`. The UI surfaces structured error messages from `DashboardDataService`.
- **Missing archives:** The service logs a warning (via structured logging) when `data/dashboard/archive/` is absent; archive support is optional.
- **New fields not showing:** Verify the CSV columns, then inspect the component renderers to ensure they check for the new field and handle nulls.
- **Historical Performance shows "No matching games":**
  - **Root Cause:** Game ID mismatch between predictions and results
  - **Check:** Verify predictions use 7-digit game_ids (e.g., "2500026") not 8-digit (e.g., "22500026")
  - **Fix Location:** [src/nba_app/inference/main.py:373-375](../src/nba_app/inference/main.py#L373-L375) normalizes game_id
- **Team names showing abbreviations (ATL, BOS):**
  - **Root Cause:** Team mapping not loaded or converted properly
  - **Check:** Verify `configs/nba/team_mapping.yaml` has `abbrev_to_full_name` section
  - **Fix:** Use `vars(config.abbrev_to_full_name)` to convert SimpleNamespace to dict
- **Streamlit deprecation warnings:**
  - Use `update_yaxes()` instead of `update_yaxis()`
  - Add `observed=False` to pandas groupby operations with categorical columns
  - Avoid passing deprecated Plotly config keyword arguments

For deeper debugging, enable Streamlit's developer tools (`?` menu → "Developer options") or tail the application log configured by the DI container.

## Recent Changes

### 2025-11-08: Team Drilldown Redesign with Pre-Calculated Metrics

**Architecture Change - Single Source of Truth:**
- Complete redesign of Team Drilldown page to use pre-calculated team_performance data from dashboard prep
- Eliminated all UI-side calculations for accuracy metrics
- Fixed home/away accuracy bug that was showing incorrect 100% values for several teams
- Lowered `min_games` threshold from 10 to 5 for early season (will increase as more data accumulates)

**Team Performance Data Pipeline:**
- Fixed `team_performance_analyzer.py` to use `predicted_winner_prob` column instead of missing `predicted_probability`
- Pipeline now generates team_performance section with 30 teams (with min 5 games each)
- Added home/away accuracy split, total predictions, and average win probability per team

**New Team Drilldown Features:**
- **Team Performance Summary:** Clean 4-metric header showing total games, overall accuracy, avg win probability, home game percentage
- **Home vs Away Performance:** Bar chart with actual pre-calculated accuracy values (no more 100% bugs)
- **Recent Performance Trend:** Improved visualization replacing old prediction timeline
  - Dual-axis chart with win/loss bars (green/red), prediction correctness line (blue), and win probability curve (orange)
  - Shows last 20 games in chronological order
  - Much more actionable than previous timeline mixing home/away contexts
- **Recent Games Detail:** Configurable table (5-50 games) with full game details

**Key Files Modified:**
- `streamlit_app/pages/2_Team_Drilldown.py` - Complete rewrite following single-source-of-truth principle
- `src/nba_app/dashboard_prep/team_performance_analyzer.py` - Fixed column reference bug
- `configs/nba/dashboard_prep_config.yaml` - Lowered min_games from 10 to 5
- `docs/streamlit_dashboard_reference.md` - Added Team Drilldown documentation

**Design Principles Applied:**
- All calculations in dashboard prep, not UI
- No backward compatibility with old buggy calculations
- Clean separation: pipeline computes, UI visualizes
- Configuration-driven (min_games threshold in config)

### 2025-11-07: Terminology Standardization - "Confidence" → "Predicted Probability"

**Rationale:**
- **Professional ML terminology**: Replaced ambiguous "confidence" with standard ML term "predicted probability"
- **Clarity for ML practitioners**: Makes it clear we're referring to calibrated model output probabilities
- **Avoids confusion**: "Confidence" could mean prediction intervals, calibration quality, or other concepts

**Changes Made:**
- **Configuration**: Updated `dashboard_prep_config.yaml`
  - Column: `confidence` → `predicted_probability` (calibrated probability of predicted winner)
  - Metric: `avg_confidence` → `avg_predicted_probability`
- **Backend (dashboard_prep)**: Updated all Python modules
  - `predictions_aggregator.py`: Uses `predicted_probability` column
  - `performance_calculator.py`: Metric name changed to `avg_predicted_probability`
  - `team_performance_analyzer.py`: Metrics use `avg_predicted_probability`
- **Streamlit Components**: Updated all UI components
  - `kpi_panel.py`: Uses `predicted_probability` column
  - `matchup_cards.py`: Filtering and sorting on `predicted_probability`
  - `calibration_charts.py`: Chart titles reference "predicted probability threshold"
- **Streamlit Pages**: Updated all page files
  - `0_Todays_Predictions.py`: Uses `get_high_probability_threshold()`
  - `1_Historical_Performance.py`: Chart titles and descriptions updated
  - `2_Model_Analysis.py`: Technical explanations use "predicted probability"
  - `2_Team_Drilldown.py`: Column references updated
- **Services**: Updated helper functions
  - `data_loader.py`: `get_high_confidence_threshold()` → `get_high_probability_threshold()`
  - `dashboard_data_service.py`: Method renamed to `get_high_probability_threshold()`
- **Documentation**: This reference guide updated throughout

**Display Terminology:**
- UI labels use "Win Probability" or "Predicted Probability" (user-friendly)
- Technical docs use "Predicted Probability (calibrated)" (precise)
- Code uses `predicted_probability` (clear variable naming)

### 2025-11-07: Historical Performance and Model Analysis Refactoring

**Architecture Change - Single Source of Truth:**
- Moved all metric calculations from Streamlit to dashboard prep pipeline
- Streamlit now uses pre-calculated `prediction_correct`, `predicted_winner_won`, and drift metrics
- Enables use of other BI tools (Power BI, Tableau) with same authoritative dashboard data
- Eliminated calculation discrepancies between pages

**Historical Performance Page Overhaul:**
- **Removed global filter** - Each chart now has its own time period filter for independent analysis
- **Daily Performance Section:**
  - Added individual filter: 7/14/30/All days (default: All)
  - Shows daily accuracy, season running accuracy, period average, and season average baseline
  - Increased from 10 days to 365 days of historical data (`lookback_days: 365`)
- **Threshold Analysis Section:**
  - Added individual filter: 14/30/All days (default: All)
  - Shows error rate vs confidence threshold with key threshold summary
- **Team Accuracy Section (NEW):**
  - Added individual filter: 14/30/All days (default: All)
  - Horizontal bar chart ranking teams from worst to best accuracy
  - Color-coded: green (≥70%), orange (60-70%), red (<60%)
  - Minimum 5 games per team required
- **Recent Games Detail Section (NEW):**
  - Game-by-game table with conditional formatting
  - Error highlighting by confidence level: light red (<60%), orange (60-70%), dark red (≥70%)
  - Correct predictions shown in green
  - Filterable by error severity with configurable max games display
- **Removed calibration curve** - Moved to new Model Analysis page for technical users
- **Added navigation** - Jump links to each section within page
- **Cross-page link** - Added link to Model Analysis page for technical diagnostics

**Model Analysis Page (NEW):**
- **Purpose:** Technical model diagnostics for ML practitioners
- **Calibration Diagnostics Section:**
  - 20-bin calibration curve (50-100% range for normalized probabilities)
  - Combined sharpness distribution + calibration line chart
  - Overall Brier score, average predicted probability, and actual win rate metrics
  - Bin-level statistics table with games, avg predicted, actual rate, Brier score
  - Individual filter: 14/30/All days (default: All)
- **Drift Monitoring Section:**
  - Accuracy over time with 7-day rolling average
  - Uses pre-calculated drift data from dashboard prep pipeline
  - Optional Brier score and calibration error trend charts (if available in data)
  - Drift summary statistics: mean accuracy, std dev, worst day, best day
  - Individual filter: 14/30/All days (default: All)
- **Error Analysis Section:**
  - Placeholder with high-level metrics
  - Framework for future detailed error pattern analysis
- **Navigation:** Jump links to each section for easy page navigation

**Configuration Changes:**
- Updated `dashboard_prep_config.yaml`:
  - `results.lookback_days: 365` (increased from 10 to include entire season)
  - Added `predicted_winner_prob` and `predicted_winner_won` to results columns
  - Enabled drift monitoring section with daily metrics

**New Component:**
- Created `streamlit_app/components/calibration_charts.py`:
  - `prepare_calibration_bins()` - Bins predictions with 50%+ probability into 20 bins
  - `calculate_threshold_metrics()` - Computes error rate and accuracy at each confidence threshold
  - `create_calibration_bin_chart()` - Combined sharpness + calibration visualization
  - `create_threshold_error_chart()` - Error rate vs threshold chart
  - `calculate_brier_score()` - Brier score calculation for model evaluation

**Pipeline Changes:**
- Updated `results_aggregator.py`:
  - Added `predicted_winner_prob` normalization (always ≥50%)
  - Added `predicted_winner_won` binary outcome
  - Pre-calculates all metrics used by Streamlit

**Bug Fixes:**
- Fixed accuracy discrepancy: Predictions page showed 68.0% while Historical Performance showed 71.6%
  - Root cause: Results section only had 10 days (67 games, 71.6%) vs metrics section with all games (97 games, 68.0%)
  - Fix: Increased lookback_days to 365 so both pages use consistent data
- Fixed 7-day filter showing entire season instead of 7 days
  - Added post-merge filtering to ensure only filtered dates are displayed

**Key Files Modified:**
- `streamlit_app/pages/1_Historical_Performance.py` - Major refactor with individual filters and new sections
- `streamlit_app/pages/2_Model_Analysis.py` - New technical diagnostics page
- `streamlit_app/components/calibration_charts.py` - New calibration analysis component
- `src/nba_app/dashboard_prep/results_aggregator.py` - Added probability normalization
- `configs/nba/dashboard_prep_config.yaml` - Increased lookback, added new columns

### 2025-11-04: Streamlit UI Overhaul

**Navigation and Structure:**
- Restructured app to use page-based navigation instead of single-page layout
- Main `app.py` now auto-redirects to `0_Todays_Predictions.py`
- Created dedicated page: `pages/0_Todays_Predictions.py` (main predictions view)
- Hidden main "app" entry from navigation menu across all pages
- Added "NBA Predictions and Analysis" title above navigation menu using CSS

**Today's Predictions Page:**
- Changed main title from "NBA Predictions Control Panel" to "Today's Predictions"
- Added date subtitle in format "Tuesday November 04, 2025" below title
- Simplified header KPIs from 5 to 4 columns (removed redundant Prediction Date)
- Renamed "Games to Predict" metric to "Games Today" for clarity
- Added horizontal separator between header and metrics

**Sidebar Improvements:**
- Removed Archive dropdown (moved to Historical Performance page conceptually)
- Removed Date filter (unnecessary with today-only focus)
- Removed Team filter (unnecessary with ~7-8 games typical)
- Kept only Win Probability Threshold slider (0.35 default)
- Added helpful caption directing users to Historical Performance page

**Matchup Cards:**
- Cards now sorted by win probability (confidence) - highest first
- Improved sorting logic to use `confidence` column primarily

**Key Files Modified:**
- `streamlit_app/app.py` - Simplified to auto-redirect landing page
- `streamlit_app/pages/0_Todays_Predictions.py` - New main predictions page with updated header
- `streamlit_app/pages/1_Historical_Performance.py` - Added navigation styling
- `streamlit_app/pages/2_Team_Drilldown.py` - Added navigation styling
- `streamlit_app/components/matchup_cards.py` - Simplified filter panel, improved sorting

### 2025-11-01: Phase 3.5 Updates

**Data Integration:**
- Fixed game ID mismatch preventing historical data display (8-digit → 7-digit normalization)
- Increased historical visibility from 8 to 57 games (`lookback_days: 1` → `10`)
- Added header KPI metrics (Season Accuracy, 7-Day Accuracy, Total Games, etc.)
- Integrated full team names from `team_mapping.yaml` throughout UI
- Changed matchup format from "Home vs Away" to "Away @ Home" with full names
- Win probability now shows predicted winner's probability (not always home team)
- Fixed all Streamlit and Plotly deprecation warnings
- Implemented proper section filtering (predictions/results/metrics separation)

**Key Files Modified:**
- `src/nba_app/inference/main.py` - Game ID normalization
- `src/nba_app/dashboard_prep/results_aggregator.py` - Team name preservation
- `configs/nba/dashboard_prep_config.yaml` - Increased lookback_days
- `configs/nba/team_mapping.yaml` - Added full team name mappings
