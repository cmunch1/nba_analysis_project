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

- **Filter panel (sidebar):** Drives matchup filtering by date, team, and probability threshold. Default threshold uses the configured `dashboard_prep.predictions.high_probability_threshold`.
- **KPI panel:** Shows total games, high-probability counts, and average win probability for the filtered set.
- **Matchup grid:** Card-based layout highlighting predicted winner (team name), home win probability, and reliability chips ("High Win Probability", "Recent Accuracy").
- **Dataset preview:** Expandable table for quick inspection of the raw dashboard output in its current state.

Because the filters, KPIs, and cards operate on the same data frame, you only need to add new columns to the nightly CSV and reference them in the component renderers to surface new information.

## Components

- `matchup_cards.render_filter_panel`: Sidebar UI; update here when introducing new filters (e.g., head-to-head toggle).
- `matchup_cards.render_matchup_grid`: Controls card layout and reliability chips. Extend this function to display injuries, implied odds, or model-version metadata.
- `kpi_panel.render_kpi_panel`: Summary metrics for the filtered dataset. Add additional KPIs here (e.g., coverage rate) and ensure null-safe calculations.

All components tolerate missing columns by checking their presence before rendering metrics to keep the UI resilient while the pipeline evolves.

## Supporting Pages

Streamlit automatically exposes files under `streamlit_app/pages/` as additional tabs.

- **Historical Performance:** Calibration scatter plot, daily accuracy trend, and summary stats. Requires columns for `game_date`, calibrated probabilities, and actual outcomes.
- **Team Drilldown:** Per-team confidence timeline, home/away accuracy bars, and recent games table. Works with `home_team`, `away_team`, `predicted_winner`, optional `actual_winner`, and probability columns.

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

For deeper debugging, enable Streamlit’s developer tools (`?` menu → “Developer options”) or tail the application log configured by the DI container.***
