# STILL UNDER CONSTRUCTION

Do not use this project yet. Significant changes are ongoing.

Docker images and GitHub Actions workflows are not working correctly at this time.




# NBA Win Prediction System 🏀

An end-to-end machine learning system for predicting NBA game outcomes with ~70% accuracy. Features automated data collection, feature engineering, calibrated predictions, and an interactive dashboard.



**Public Datasets:**
- [NBA Game Statistics (Daily Updated)](https://kaggle.com/datasets/YOUR_KAGGLE_USERNAME/nba-game-stats-daily)
- [NBA Processed Data](https://kaggle.com/datasets/YOUR_KAGGLE_USERNAME/nba-processed-data)

## 📊 Live Dashboard

View daily predictions and historical performance:
- **Today's Predictions**: Win probabilities for upcoming games
- **Historical Performance**: Model accuracy tracking over time
- **Team Analysis**: Team-by-team prediction performance
- **Model Diagnostics**: Calibration curves and drift monitoring

## ✨ Key Features

### Production ML Pipeline
- ✅ **Automated Data Collection**: Nightly scraping from NBA.com (3am EST)
- ✅ **Feature Engineering**: 849 features with rolling averages, streaks, ELO ratings
- ✅ **Calibrated Predictions**: Isotonic regression for reliable win probabilities
- ✅ **Prediction Backfill**: Automatically fills missed predictions for up to 14 days
- ✅ **Performance Tracking**: Daily accuracy metrics and model drift monitoring

### Software Engineering
- ✅ **Docker Containerized**: Consistent environment everywhere (CPU + GPU support)
- ✅ **VS Code Dev Containers**: Interactive development inside containers
- ✅ **Modular Workflows**: Separate data collection, ML pipeline, and dashboard
- ✅ **Dependency Injection**: Clean, testable architecture throughout
- ✅ **Comprehensive Logging**: Structured logs for debugging and monitoring
- ✅ **MLflow Integration**: Model versioning, registry, and experiment tracking

### Data Architecture
- ✅ **Kaggle as Data Store**: Free, unlimited, versioned datasets
- ✅ **Multiple Data Sources**: Kaggle, local files, or scrape fresh data
- ✅ **Deterministic Features**: Reproducible feature ordering for inference
- ✅ **Feature Schema Validation**: Automatic feature reordering at inference time

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Webscraping    │ ───> │ Data Processing  │ ───> │Feature Engineer │
│  (NBA.com)      │      │ (Consolidation)  │      │(849 features)   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                             │
┌─────────────────┐      ┌──────────────────┐              │
│   Dashboard     │ <─── │   Inference      │ <────────────┘
│  (Streamlit)    │      │ (XGBoost + Cal.) │
└─────────────────┘      └──────────────────┘
         │                        │
         └────────────────────────┴───> Kaggle (Data Store)
```

### GitHub Actions Workflows

1. **Data Collection** (`data_collection.yml`) - Nightly at 3am EST
   - Scrapes NBA.com for schedule and results
   - Updates Kaggle datasets automatically
   - **Maintainer only** (requires proxy + secrets)

2. **Inference with Kaggle Data** (`inference_with_kaggle_data.yml`) - Nightly at 4am EST
   - Downloads from Kaggle (public, no secrets!)
   - Generates predictions with uncertainty
   - Updates dashboard statistics
   - **Anyone can fork and run**

3. **Docker Build** (`docker-build.yml`) - On code changes
   - Builds and pushes to GitHub Container Registry
   - Used by all workflows for consistency

4. **Local Development** - Manual trigger
   - Test with Kaggle, local, or scraped data
   - Flexible for experimentation

## 📦 Project Structure

```
├── src/
│   ├── nba_app/              # NBA-specific application code
│   │   ├── webscraping/      # Data collection from NBA.com
│   │   ├── data_processing/  # Clean and consolidate data
│   │   ├── feature_engineering/  # 849 NBA-specific features
│   │   ├── inference/        # Generate predictions
│   │   └── dashboard_prep/   # Prepare data for dashboard
│   └── ml_framework/         # Reusable ML framework
│       ├── core/             # Config, logging, error handling
│       ├── model_testing/    # Training and evaluation
│       ├── preprocessing/    # Model-aware preprocessing
│       ├── postprocessing/   # Calibration and uncertainty
│       ├── model_registry/   # MLflow integration
│       └── visualization/    # Chart generation
├── streamlit_app/            # Interactive dashboard
├── configs/                  # YAML configuration files
├── scripts/                  # Helper scripts and pipeline
├── .github/workflows/        # CI/CD automation
├── Dockerfile               # Pipeline container
└── Dockerfile.streamlit     # Dashboard container
```

## 🎯 Model Performance

**Current Production Model (v5):**
- **Algorithm**: XGBoost with isotonic calibration
- **Cross-Validation AUC**: ~0.70
- **Accuracy**: ~68% (95/139 games correct as of Nov 2025)
- **Calibration**: Brier score ~0.22
- **Features**: 849 engineered features (from 1,764 initial)

**Feature Categories:**
- Traditional stats (points, rebounds, assists)
- Advanced stats (True Shooting %, Usage Rate)
- Four Factors (shooting, turnovers, rebounding, free throws)
- Rolling averages (3, 5, 10, 15, 20, 40 game windows)
- Opponent-adjusted stats
- ELO ratings (with 100-point home advantage)
- Streaks and trends

## 📚 Documentation

### Getting Started
- **[DOCKER_SETUP_SUMMARY.md](DOCKER_SETUP_SUMMARY.md)** - Complete Docker setup guide
- **[docs/DOCKER.md](docs/DOCKER.md)** - Docker deployment reference
- **[docs/GPU_SUPPORT.md](docs/GPU_SUPPORT.md)** - GPU acceleration guide
- **[DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md)** - Full deployment roadmap

### For Contributors
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[.devcontainer/README.md](.devcontainer/README.md)** - VS Code Dev Containers guide
- **[docs/AI/core_framework_usage.md](docs/AI/core_framework_usage.md)** - Framework patterns and DI
- **[docs/AI/interfaces.md](docs/AI/interfaces.md)** - Abstract interfaces
- **[docs/streamlit_dashboard_reference.md](docs/streamlit_dashboard_reference.md)** - Dashboard guide

### Technical Reference
- **[docs/AI/config_reference.tree](docs/AI/config_reference.tree)** - Configuration hierarchy
- **[docs/AI/directory_tree.txt](docs/AI/directory_tree.txt)** - Project structure
- **[scripts/README.md](scripts/README.md)** - Pipeline script usage


