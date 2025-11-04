# Streamlit Dashboard Reference Guide

This guide summarizes the Streamlit implementation that powers the NBA predictions control panel. Use it when running the app locally, extending UI components, or wiring data from the nightly pipeline.

## Project Layout

```
streamlit_app/
├── app.py                      # Landing page (matchup filters, KPIs, cards)
├── data_loader.py              # Cached data access helpers (DI-backed)
├── di_container.py             # Streamlit-specific dependency container
├── README.md                   # Quick start instructions
├── components/
│   ├── __init__.py
│   ├── kpi_panel.py            # Summary metric display
│   └── matchup_cards.py        # Filter controls + matchup grid rendering
└── pages/
    ├── 1_Historical_Performance.py
    └── 2_Team_Drilldown.py
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
  - **Win Probability:** Displays the predicted winner's probability (not always home team)
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
  - Extend this function to display injuries, implied odds, or model-version metadata
- `kpi_panel.render_kpi_panel`: Summary metrics for the filtered dataset. Add additional KPIs here (e.g., coverage rate) and ensure null-safe calculations.

All components tolerate missing columns by checking their presence before rendering metrics to keep the UI resilient while the pipeline evolves.

## Supporting Pages

Streamlit automatically exposes files under `streamlit_app/pages/` as additional tabs.

- **Historical Performance (1_Historical_Performance.py):**
  - **Section Filtering:** Automatically filters to show only 'results' section (games with actual outcomes)
  - **Calibration Curve:** Scatter plot comparing predicted probabilities to observed win rates
  - **Daily Accuracy Trend:** Line chart showing accuracy over time with game counts
  - **Metric Snapshot:** Summary statistics (games evaluated, overall accuracy, avg win probability)
  - **Recent Updates:**
    - Fixed deprecation warnings (`update_yaxis` → `update_yaxes`, added `observed=False` to groupby)
    - Now displays all historical games (increased from 8 to 57 via `lookback_days: 10` config)
  - **Required Columns:** `game_date`, `calibrated_home_win_prob`, `actual_home_score`, `actual_away_score` or `actual_winner`

- **Team Drilldown (2_Team_Drilldown.py):**
  - Per-team confidence timeline, home/away accuracy bars, and recent games table
  - Works with `home_team`, `away_team`, `predicted_winner`, optional `actual_winner`, and probability columns

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
