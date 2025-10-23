# NBA Analysis Project - Deployment Plan

## Executive Summary

This document outlines the plan to complete deployment of the NBA win prediction batch pipeline. The system runs nightly at 3am EST to:
1. Scrape today's game schedule and yesterday's results (with proxy)
2. Generate predictions for upcoming games with uncertainty quantification
3. Validate yesterday's predictions against actual results
4. Create a dashboard-optimized CSV with predictions, results, and performance metrics

## Current Status

### ✅ Completed Components

1. **Feature Engineering Pipeline** ([src/nba_app/feature_engineering/](src/nba_app/feature_engineering/))
   - ✅ Loads historical data and today's matchups
   - ✅ Creates placeholder rows for upcoming games (stats=0)
   - ✅ Engineers 849 features based on allowlist
   - ✅ Preserves metadata columns for dashboard (h_team, v_team, h_match_up, v_match_up)
   - ✅ Outputs: `data/engineered/engineered_features.csv` (856 columns: 7 metadata + 849 features)
   - **Config**: [configs/nba/feature_engineering_config.yaml](configs/nba/feature_engineering_config.yaml)

2. **Feature Allowlist Filtering**
   - ✅ Loads 849 features from `ml_artifacts/features/allowlists/feature_allowlist_latest.yaml`
   - ✅ Filters engineered output to only include allowlisted features
   - ✅ Preserves required metadata columns (game_id, game_date, team names, target)
   - ✅ Reduces from 1,764 calculated features to 849 (51.7% reduction)
   - **Code**: [src/nba_app/feature_engineering/feature_engineer.py:746-822](src/nba_app/feature_engineering/feature_engineer.py#L746-L822)

3. **Matchup Processor**
   - ✅ Loads today's matchups from webscraping output
   - ✅ Converts team IDs to abbreviations using team_mapping.yaml
   - ✅ Creates placeholder rows matching exact schema of processed data
   - **Code**: [src/nba_app/feature_engineering/matchup_processor.py](src/nba_app/feature_engineering/matchup_processor.py)

4. **Dashboard Prep Module Structure** ([src/nba_app/dashboard_prep/](src/nba_app/dashboard_prep/))
   - ✅ Module created with 7 components
   - ⏳ Not yet tested end-to-end
   - **Config**: [configs/nba/dashboard_prep_config.yaml](configs/nba/dashboard_prep_config.yaml)

5. **Inference Config**
   - ✅ Configuration created for production pipeline
   - ⏳ Inference module not yet implemented
   - **Config**: [configs/nba/inference_config.yaml](configs/nba/inference_config.yaml)

### ⏳ Remaining Work

1. **Inference Module** - Load engineered data, filter to today's games, predict with postprocessing
2. **Artifact Loading from MLflow** - Load model, calibrator, conformal predictor from MLflow registry
3. **Dashboard Prep Testing** - Test complete data aggregation pipeline
4. **End-to-End Pipeline Test** - Run all 5 stages sequentially
5. **Docker Deployment Setup** - Create Dockerfile and deployment configs

## Architecture Decisions

### Artifact Management Strategy (Dev vs Prod)

**Development (Current - Keep This):**
```
ml_artifacts/
├── models/
│   ├── checkpoints/     # Local training experiments (NOT in git)
│   └── final/          # Local production candidates (NOT in git)
├── features/
│   ├── allowlists/     # ✅ Committed to git (YAML configs)
│   └── schemas/        # ✅ Committed to git (JSON documentation)
└── experiments/
    └── hyperparameters/ # ✅ Committed to git (baseline configs)
```

**Production (Load from MLflow):**
```
MLflow Registry ("nba_win_predictor/Production"):
├── Model artifact (XGBoost .pkl)
├── feature_allowlist.yaml (logged during training)
├── calibrator.pkl (from postprocessing)
├── conformal_predictor.pkl (from postprocessing)
└── Metrics/params (training run metadata)
```

**Rationale:**
- ✅ Fast local iteration (no network I/O)
- ✅ MLflow provides versioning for deployment
- ✅ Easy rollback (switch model versions)
- ✅ Lineage tracking (trace predictions to training run)

**Config Already Set:**
```yaml
# configs/core/postprocessing_config.yaml
calibration:
  save_to_registry: true  # ✅ Artifacts saved to MLflow

conformal:
  save_to_registry: true  # ✅ Artifacts saved to MLflow
```

### Feature Engineering Workflow

**Critical Design Pattern (from user feedback):**

The user's proven workflow from a previous project:
1. **Webscraping** (3am EST) saves:
   - `data/newly_scraped/todays_matchups.csv` (scheduled games)
   - `data/newly_scraped/todays_games_ids.csv` (game IDs)

2. **Feature Engineering** loads matchups and:
   - Creates placeholder rows (stats=0) for today's games
   - Concatenates with historical processed data
   - Engineers features using rolling windows (only historical data used for today's features)
   - Output includes today's games WITH engineered features

3. **Inference** (simplified):
   - Loads `data/engineered/engineered_features.csv`
   - Filters to today's game IDs
   - Predicts (preprocessing automatic via ModelPredictor)
   - Applies postprocessing (calibration + conformal prediction)

**Key Insight:** Feature engineering handles today's matchups; inference just filters and predicts.

### Metadata Columns Decision

**Decision:** Keep human-readable team identifiers through entire pipeline

**Columns Preserved:**
- `game_id` - Unique game identifier
- `game_date` - Game date
- `h_team` - Home team abbreviation (e.g., "MIA")
- `v_team` - Away team abbreviation (e.g., "CHI")
- `h_match_up` - Home matchup description (e.g., "MIA vs. CHI")
- `v_match_up` - Away matchup description (e.g., "CHI @ MIA")
- `h_is_win` - Target variable

**Rationale:**
- Dashboard prep becomes simpler (no reconstruction needed)
- Better debugging (can see which teams each row represents)
- No model impact (excluded via `non_useful_columns` config)
- Minimal cost (4 extra string columns)

## Deployment Plan

### Phase 0: MLflow Model Registry Integration (COMPLETED ✅)

**Priority:** HIGH
**Status:** COMPLETED 2025-10-23

**Completed Tasks:**

1. ✅ **Enhanced MLflow Model Registry** ([src/ml_framework/model_registry/mlflow_model_registry.py](src/ml_framework/model_registry/mlflow_model_registry.py))
   - Added `additional_artifacts` parameter to `save_model()` method
   - Saves calibrator and conformal_predictor as pickle files
   - Loads additional artifacts in `load_model()` method with error handling

2. ✅ **Integrated Model Registry into DI Container**
   - Added `model_registry` to [src/ml_framework/core/common_di_container.py](src/ml_framework/core/common_di_container.py)
   - Injected into ModelTester via [src/ml_framework/model_testing/di_container.py](src/ml_framework/model_testing/di_container.py)

3. ✅ **Created Configuration-Based Model Save Workflow**
   - Added registry configuration to [configs/core/model_testing_config.yaml](configs/core/model_testing_config.yaml)
   - Created `_save_model_to_registry()` helper function in [src/ml_framework/model_testing/main.py](src/ml_framework/model_testing/main.py)
   - Only saves after validation set testing (not OOF)
   - Respects `save_model_to_registry: true/false` flag

4. ✅ **Fixed Artifact Attribute Access Bug**
   - Fixed AttributeError when accessing `.method` from calibration/conformal artifacts
   - Changed from `.get('method')` to `hasattr()` check + direct attribute access

5. ✅ **Verified Complete Artifact Save**
   - Tested model training + registry save workflow
   - All artifacts successfully saved:
     - Model: XGBoost model registered as "nba_win_predictor" version 1
     - Preprocessor: 45 KB (`preprocessor_artifact.pkl`)
     - Calibrator: 27 KB (`calibrator_artifact.pkl`)
     - Conformal Predictor: 27 KB (`conformal_predictor_artifact.pkl`)

**Known Issues:**
- ⚠️ Calibration optimization has an indexing bug (documented in [CALIBRATION_OPTIMIZATION_BUG.md](CALIBRATION_OPTIMIZATION_BUG.md))
- **Workaround**: Set `calibration.optimize: false` in [configs/core/postprocessing_config.yaml](configs/core/postprocessing_config.yaml)
- Manual calibration method (sigmoid) works correctly

**Artifacts Location:**
```
mlruns/209835064394409033/6d263e086b924de986ab700a17e9e9a7/artifacts/artifacts/
├── preprocessor_artifact.pkl (45 KB)
├── calibrator_artifact.pkl (27 KB)
└── conformal_predictor_artifact.pkl (27 KB)

mlruns/models/nba_win_predictor/version-1/
└── meta.yaml (points to run 6d263e086b924de986ab700a17e9e9a7)
```

### Phase 1: Complete Inference Module

**Priority:** HIGH
**Status:** Ready to Start

**Tasks:**

1. **Create inference module structure:**
   ```
   src/nba_app/inference/
   ├── __init__.py
   ├── main.py              # CLI entry point
   ├── inference_engine.py  # Core prediction logic
   └── di_container.py      # Dependency injection
   ```

2. **Implement ArtifactLoader utility:**
   ```python
   # src/ml_framework/inference/artifact_loader.py
   class ArtifactLoader:
       """
       Load artifacts from MLflow in production, local files in development.
       Implements fallback strategy for resilience.
       """

       def load_model(self, model_uri: str) -> Any:
           """Load model from MLflow Model Registry"""

       def load_postprocessing_artifacts(self, run_id: str) -> Dict:
           """Load calibrator + conformal predictor from training run"""

       def load_feature_allowlist(self, run_id: str) -> Set[str]:
           """Load feature allowlist from training run"""
   ```

3. **Inference workflow:**
   ```python
   # Simplified pipeline (200-300 lines)
   def run_inference():
       # 1. Load today's game IDs
       todays_game_ids = load_game_ids("data/newly_scraped/todays_games_ids.csv")

       # 2. Load engineered data (already has today's games with features)
       engineered_df = load_dataframe("data/engineered/engineered_features.csv")

       # 3. Filter to today's games
       todays_features = engineered_df[engineered_df['game_id'].isin(todays_game_ids)]

       # 4. Load model + artifacts from MLflow
       model_uri = "models:/nba_win_predictor/Production"
       model = artifact_loader.load_model(model_uri)
       artifacts = artifact_loader.load_postprocessing_artifacts(run_id)

       # 5. Predict (preprocessing automatic via ModelPredictor)
       raw_predictions = model_predictor.predict(todays_features)

       # 6. Apply postprocessing
       calibrated_probs = calibrator.transform(raw_predictions)
       prediction_sets = conformal_predictor.predict(calibrated_probs)

       # 7. Save predictions
       save_predictions("data/predictions/predictions_{date}.csv")
   ```

4. **Config already exists:**
   - ✅ [configs/nba/inference_config.yaml](configs/nba/inference_config.yaml)
   - Specifies: model identifier, input paths, postprocessing settings, output format

**Files to Create:**
- `src/nba_app/inference/main.py`
- `src/nba_app/inference/inference_engine.py`
- `src/nba_app/inference/di_container.py`
- `src/ml_framework/inference/artifact_loader.py` (if not exists)

**Files to Modify:**
- None (inference is net-new)

**Testing:**
```bash
# Test with today's engineered data
uv run -m src.nba_app.inference.main

# Expected output: data/predictions/predictions_2025-10-23.csv
```

### Phase 2: Test Dashboard Prep Module

**Priority:** HIGH
**Status:** Module Created, Not Tested

**Tasks:**

1. **Verify module structure:**
   ```
   src/nba_app/dashboard_prep/
   ├── __init__.py                  ✅ Created
   ├── main.py                      ✅ Created
   ├── dashboard_data_generator.py  ✅ Created
   ├── predictions_aggregator.py    ✅ Created
   ├── results_aggregator.py        ✅ Created
   ├── performance_calculator.py    ✅ Created
   ├── team_performance_analyzer.py ✅ Created
   └── di_container.py              ✅ Created
   ```

2. **Test complete pipeline:**
   ```bash
   # Run dashboard prep (requires predictions + actual results)
   uv run -m src.nba_app.dashboard_prep.main

   # Expected output: data/dashboard/dashboard_data.csv
   ```

3. **Validate dashboard output sections:**
   - **predictions**: Tomorrow's games with uncertainty
   - **results**: Yesterday's games with validation
   - **metrics**: Season/7day/30day performance
   - **team_performance**: Per-team accuracy
   - **drift**: Daily metrics for monitoring

**Config:**
- ✅ [configs/nba/dashboard_prep_config.yaml](configs/nba/dashboard_prep_config.yaml)

**Dependencies:**
- Requires: predictions from inference module
- Requires: actual results from webscraping (yesterday's box scores)

### Phase 3: End-to-End Pipeline Testing

**Priority:** MEDIUM
**Status:** Not Started

**Tasks:**

1. **Create pipeline orchestration script:**
   ```bash
   # scripts/run_nightly_pipeline.sh
   #!/bin/bash
   set -e  # Exit on error

   echo "=== Stage 1: Webscraping ==="
   uv run -m src.nba_app.webscraping.main

   echo "=== Stage 2: Data Processing ==="
   uv run -m src.nba_app.data_processing.main

   echo "=== Stage 3: Feature Engineering ==="
   uv run -m src.nba_app.feature_engineering.main

   echo "=== Stage 4: Inference ==="
   uv run -m src.nba_app.inference.main

   echo "=== Stage 5: Dashboard Prep ==="
   uv run -m src.nba_app.dashboard_prep.main

   echo "=== Pipeline Complete ==="
   ```

2. **Test with real data:**
   ```bash
   # Run complete pipeline
   ./scripts/run_nightly_pipeline.sh

   # Verify outputs:
   ls data/newly_scraped/todays_matchups.csv
   ls data/processed/teams_boxscores.csv
   ls data/engineered/engineered_features.csv
   ls data/predictions/predictions_*.csv
   ls data/dashboard/dashboard_data.csv
   ```

3. **Add error handling and logging:**
   - Pipeline should log to `logs/pipeline_{date}.log`
   - Email/Slack alerts on failure (optional)

### Phase 4: Docker Containerization

**Priority:** MEDIUM
**Status:** Not Started

**Tasks:**

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.11-slim

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc g++ \
       chromium chromium-driver \
       && rm -rf /var/lib/apt/lists/*

   # Install uv
   RUN pip install uv

   # Copy application code
   WORKDIR /app
   COPY pyproject.toml uv.lock ./
   COPY src/ ./src/
   COPY configs/ ./configs/

   # Install Python dependencies
   RUN uv sync --frozen

   # Environment variables (set at runtime)
   ENV MLFLOW_TRACKING_URI=""
   ENV PROXY_URL=""

   # Entry point
   CMD ["uv", "run", "scripts/run_nightly_pipeline.sh"]
   ```

2. **Create docker-compose.yml for local testing:**
   ```yaml
   version: '3.8'
   services:
     nba_pipeline:
       build: .
       environment:
         - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
         - PROXY_URL=${PROXY_URL}
       volumes:
         - ./data:/app/data
         - ./logs:/app/logs
       networks:
         - nba_network

   networks:
     nba_network:
       driver: bridge
   ```

3. **Test container locally:**
   ```bash
   # Build
   docker build -t nba-pipeline:latest .

   # Run
   docker run --rm \
     -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
     -e PROXY_URL=$PROXY_URL \
     -v $(pwd)/data:/app/data \
     nba-pipeline:latest
   ```

### Phase 5: Deployment Platform Setup

**Priority:** LOW
**Status:** Not Started

**Options:**

#### Option A: GitHub Actions (Simplest)
```yaml
# .github/workflows/nightly_pipeline.yml
name: Nightly NBA Pipeline
on:
  schedule:
    - cron: '0 8 * * *'  # 3am EST = 8am UTC
  workflow_dispatch:  # Manual trigger

jobs:
  run_pipeline:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/${{ github.repository }}/nba-pipeline:latest
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      PROXY_URL: ${{ secrets.PROXY_URL }}
    steps:
      - name: Run Pipeline
        run: uv run scripts/run_nightly_pipeline.sh
```

**Pros:**
- ✅ Simplest setup (GitHub handles scheduling)
- ✅ Free for private repos (2,000 minutes/month)
- ✅ Built-in secrets management

**Cons:**
- ⚠️ 6-hour max runtime (should be enough)
- ⚠️ No persistent storage (need S3/external DB)

#### Option B: AWS ECS Fargate (More Scalable)
```yaml
# Task Definition
{
  "family": "nba-pipeline",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [{
    "name": "nba-pipeline",
    "image": "YOUR_ECR_REPO/nba-pipeline:latest",
    "environment": [
      {"name": "MLFLOW_TRACKING_URI", "value": "https://..."}
    ],
    "secrets": [
      {"name": "PROXY_URL", "valueFrom": "arn:aws:secretsmanager:..."}
    ]
  }]
}

# EventBridge Rule (for scheduling)
aws events put-rule \
  --name nba-nightly-pipeline \
  --schedule-expression "cron(0 8 * * ? *)"
```

**Pros:**
- ✅ No time limits
- ✅ Auto-scaling if needed
- ✅ Integrates with AWS ecosystem (S3, RDS)

**Cons:**
- ⚠️ More complex setup
- ⚠️ Monthly costs ($20-50/month estimated)

#### Option C: Vercel Cron + API (UI Focus)
```javascript
// pages/api/run-pipeline.js
export default async function handler(req, res) {
  // Trigger pipeline via webhook
  // Pipeline runs on separate compute (AWS/GH Actions)
  // Vercel just serves dashboard UI
}
```

**Pros:**
- ✅ Great for serving dashboard UI
- ✅ Free tier generous

**Cons:**
- ⚠️ Not ideal for heavy compute
- ⚠️ Still need separate compute for pipeline

**Recommendation:** Start with **GitHub Actions** (simplest), migrate to AWS ECS if you need more control/resources.

## Key Files Reference

### Configuration Files
- **Feature Engineering**: [configs/nba/feature_engineering_config.yaml](configs/nba/feature_engineering_config.yaml)
- **Inference**: [configs/nba/inference_config.yaml](configs/nba/inference_config.yaml)
- **Dashboard Prep**: [configs/nba/dashboard_prep_config.yaml](configs/nba/dashboard_prep_config.yaml)
- **Postprocessing**: [configs/core/postprocessing_config.yaml](configs/core/postprocessing_config.yaml)
- **Team Mapping**: [configs/nba/team_mapping.yaml](configs/nba/team_mapping.yaml)

### Source Code
- **Feature Engineering Main**: [src/nba_app/feature_engineering/main.py](src/nba_app/feature_engineering/main.py)
- **Feature Engineer**: [src/nba_app/feature_engineering/feature_engineer.py](src/nba_app/feature_engineering/feature_engineer.py)
- **Matchup Processor**: [src/nba_app/feature_engineering/matchup_processor.py](src/nba_app/feature_engineering/matchup_processor.py)
- **Dashboard Prep Main**: [src/nba_app/dashboard_prep/main.py](src/nba_app/dashboard_prep/main.py)
- **Dashboard Data Generator**: [src/nba_app/dashboard_prep/dashboard_data_generator.py](src/nba_app/dashboard_prep/dashboard_data_generator.py)

### Artifacts
- **Feature Allowlist**: [ml_artifacts/features/allowlists/feature_allowlist_latest.yaml](ml_artifacts/features/allowlists/feature_allowlist_latest.yaml)
- **Model Registry**: MLflow - `models:/nba_win_predictor/Production`

### Data Flow
```
data/
├── newly_scraped/           # Webscraping output
│   ├── todays_matchups.csv
│   └── todays_games_ids.csv
├── processed/               # Cleaned data
│   └── teams_boxscores.csv
├── engineered/              # Feature engineering output
│   └── engineered_features.csv (856 columns: 7 metadata + 849 features)
├── predictions/             # Inference output
│   └── predictions_{date}.csv
└── dashboard/               # Dashboard prep output
    └── dashboard_data.csv
```

## Testing Checklist

### Pre-Deployment Testing

- [ ] **Feature Engineering**
  - [x] Loads historical data correctly
  - [x] Creates placeholder rows for today's games
  - [x] Engineers 849 features per allowlist
  - [x] Preserves metadata columns
  - [x] Output has 856 columns (7 metadata + 849 features)

- [ ] **Inference**
  - [ ] Loads engineered data
  - [ ] Filters to today's games
  - [ ] Loads model from MLflow
  - [ ] Loads postprocessing artifacts from MLflow
  - [ ] Produces calibrated probabilities
  - [ ] Generates prediction sets (conformal)
  - [ ] Saves predictions with metadata

- [ ] **Dashboard Prep**
  - [ ] Loads predictions
  - [ ] Loads actual results
  - [ ] Calculates performance metrics
  - [ ] Generates team performance analysis
  - [ ] Outputs complete dashboard CSV

- [ ] **End-to-End Pipeline**
  - [ ] All 5 stages run successfully
  - [ ] Data flows correctly between stages
  - [ ] Error handling works
  - [ ] Logging is comprehensive

- [ ] **Docker**
  - [ ] Container builds successfully
  - [ ] Pipeline runs in container
  - [ ] Environment variables work
  - [ ] Volumes mounted correctly

### Production Readiness

- [x] **MLflow Integration**
  - [x] Model registered to MLflow model registry
  - [x] Postprocessing artifacts saved with model (calibrator, conformal_predictor)
  - [x] Preprocessor artifact saved with model
  - [ ] Model promoted to "Production" stage (manual step after validation)
  - [ ] Feature allowlist logged with run (TODO: add to model save workflow)
  - [x] Can load artifacts by run_id

- [ ] **Monitoring**
  - [ ] Pipeline logs to structured format
  - [ ] Errors trigger alerts (email/Slack)
  - [ ] Performance metrics tracked over time
  - [ ] Distribution shift detection working

- [ ] **Security**
  - [ ] Secrets managed via environment variables
  - [ ] No credentials in code/configs
  - [ ] Proxy credentials secured
  - [ ] MLflow auth configured (if applicable)

## Environment Variables

Required for production deployment:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=https://your-mlflow-server.com
# or: MLFLOW_TRACKING_URI=databricks
# or: MLFLOW_TRACKING_URI=file:///path/to/mlruns (dev only)

# Proxy Configuration (for webscraping)
PROXY_URL=http://username:password@proxy-host:port
# or: PROXY_HOST, PROXY_PORT, PROXY_USER, PROXY_PASS (separate vars)

# Optional: Environment indicator
ENV=production  # or: development, staging

# Optional: Notification settings
ALERT_EMAIL=your-email@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## Cost Estimates

### GitHub Actions Deployment
- **Compute**: Free (2,000 minutes/month)
- **Storage**: Use external (S3, etc.)
- **Total**: ~$5-10/month (S3 + external services)

### AWS ECS Fargate Deployment
- **Compute**: Fargate (1 vCPU, 2GB RAM) = ~$30/month
- **Storage**: S3 = ~$5/month
- **MLflow**: Self-hosted on EC2 = ~$20/month
- **Total**: ~$55/month

### Minimal Setup (Just Predictions)
- **GitHub Actions**: Free compute
- **MLflow**: Databricks Community (free) or local
- **Storage**: GitHub artifacts (free, 90 day retention)
- **Total**: $0/month (limited features)

## Next Steps

**✅ Completed:**
0. ~~MLflow model registry integration with all artifacts~~ (DONE 2025-10-23)

**Immediate (Start Next):**
1. **Implement inference module with MLflow artifact loading** ← YOU ARE HERE
   - Create inference module structure
   - Implement artifact loading from MLflow
   - Test with saved model and artifacts
   - Generate predictions for today's games

2. Test dashboard_prep module end-to-end

**Short Term:**
3. Create pipeline orchestration script
4. Add comprehensive error handling and logging
5. Fix calibration optimization indexing bug (see [CALIBRATION_OPTIMIZATION_BUG.md](CALIBRATION_OPTIMIZATION_BUG.md))

**Medium Term:**
6. Docker containerization
7. GitHub Actions deployment setup
8. Production monitoring setup

**Long Term:**
9. UI dashboard development (Vercel/Next.js)
10. Model retraining automation
11. A/B testing framework for model versions

## Important Notes from Conversation

### Feature Engineering Insights
- The allowlist filtering was NOT working initially (all 1,764 features were saved)
- Fixed by adding `apply_feature_allowlist()` method called after `encode_game_date()`
- Target column changes from `is_win` to `h_is_win` after `merge_team_data()`
- Metadata columns (`h_team`, `v_team`, etc.) are explicitly preserved for dashboard use

### Artifact Management Philosophy
- **Development**: Local `ml_artifacts/` for fast iteration
- **Production**: MLflow for versioning and deployment
- **Git**: Only configs/schemas (YAML/JSON), not binaries (.pkl)
- This is the professional pattern - don't change it!

### Deployment Philosophy
- **Batch process** (not latency-sensitive) can load artifacts at startup
- **MLflow-centric**: Model + artifacts versioned together
- **Fallback strategy**: Load from local files if MLflow unavailable (dev mode)
- **Configuration over code**: Everything configurable via YAML

### Model Training Already Configured
- `save_to_registry: true` already set in postprocessing config
- Artifacts (calibrator, conformal predictor) should be logged to MLflow during training
- Feature allowlist should also be logged with each training run

## Questions to Resolve

1. **MLflow Server**: Where is MLflow hosted? (Databricks, self-hosted, local?)
2. **Proxy Service**: What proxy service for webscraping? (credentials, rate limits?)
3. **Dashboard UI**: React/Next.js on Vercel? Or simple HTML served from S3?
4. **Alerting**: Email, Slack, or other notification method?
5. **Data Retention**: How long to keep predictions? (archive old predictions?)
6. **Model Retraining**: Manual or automated trigger? How often?

## Success Criteria

**Pipeline is deployment-ready when:**
- [ ] All 5 stages run successfully in sequence
- [ ] Inference loads artifacts from MLflow (with local fallback)
- [ ] Dashboard CSV generated with all required sections
- [ ] Docker container runs pipeline successfully
- [ ] Deployed to GitHub Actions (or AWS) with scheduled trigger
- [ ] Monitoring and alerting configured
- [ ] Documentation complete (README, deployment guide)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-23
**Author**: Claude (AI Assistant)
**Status**: Ready for implementation
