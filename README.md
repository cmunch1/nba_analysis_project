# NBA Win Prediction System 🏀

[![Docker Build](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/actions/workflows/docker-build.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/actions/workflows/docker-build.yml)
[![ML Pipeline](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/actions/workflows/ml_pipeline.yml)

An end-to-end machine learning system for predicting NBA game outcomes with ~70% accuracy. Features automated data collection, feature engineering, calibrated predictions, and an interactive dashboard.

## 🚀 Quick Start

### For Users (Fastest Way)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project.git
cd nba_analysis_project

# 2. Download latest data from Kaggle (public, no auth required)
./scripts/download_kaggle_data.sh

# 3. Run predictions
./scripts/run_nightly_pipeline.sh --data-source kaggle

# 4. View interactive dashboard
streamlit run streamlit_app/app.py
```

### For Contributors/Forkers

```bash
# Automated setup wizard
./scripts/setup_fork.sh

# Or use Docker
docker-compose up nba-pipeline

# Or open in VS Code Dev Container (interactive development)
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

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

1. **Data Collection** (Nightly at 3am EST)
   - Scrapes NBA.com for schedule and results
   - Updates Kaggle datasets automatically
   - Maintainer only (requires proxy + secrets)

2. **ML Pipeline** (After data collection)
   - Downloads from Kaggle (public, no secrets!)
   - Generates predictions with uncertainty
   - Anyone can fork and run

3. **Docker Build** (On code changes)
   - Builds and pushes to GitHub Container Registry
   - Used by all workflows for consistency

4. **Local Development** (Manual trigger)
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

## 🐳 Docker Quick Reference

### Getting Started

**Option 1: Quick Start with Pre-Built Images** ⚡ (2-3 min)
```bash
# Pull and run (fastest, recommended for most users)
docker-compose pull
docker-compose up nba-pipeline
```

**Option 2: Build from Source** 🔧 (5-10 min)
```bash
# Build locally (for customization or if pull fails)
docker-compose build
docker-compose up nba-pipeline
```

### CPU Version (Default - Works Everywhere)

```bash
# Run pipeline with Kaggle data
docker-compose up nba-pipeline

# Run dashboard
docker-compose up nba-dashboard

# Pull specific pre-built image
docker pull ghcr.io/YOUR_GITHUB_USERNAME/nba_analysis_project:latest

# Run manually with pre-built image
docker run --rm \
  -v $(pwd)/data:/app/data \
  ghcr.io/YOUR_GITHUB_USERNAME/nba_analysis_project:latest
```

### GPU Version (For Local Development with NVIDIA GPU)

```bash
# Check if GPU is available and configured
./scripts/detect_gpu.sh

# Run with GPU acceleration (3-4x faster training)
docker-compose -f docker-compose.gpu.yml up nba-pipeline-gpu
```

**GPU Support**: Auto-detects NVIDIA GPU, falls back to CPU. See [docs/GPU_SUPPORT.md](docs/GPU_SUPPORT.md) for details.

## 🤝 Contributing

Contributions welcome! Three ways to help:

1. **Test and Report**: Run the pipeline, report issues
2. **Improve Models**: Experiment with features or algorithms
3. **Enhance Dashboard**: Add new visualizations or metrics

**For Forkers:**
- No scraping needed - use public Kaggle datasets
- No secrets required for ML pipeline
- Test with GitHub Actions "Local Development" workflow

## 🔐 Required Secrets (Maintainer Only)

For running the full data collection workflow:

```
KAGGLE_USERNAME      # Your Kaggle username
KAGGLE_KEY           # API key from kaggle.com/settings
KAGGLE_DATASET_ID    # your-username/nba-game-stats-daily
PROXY_URL            # Proxy for NBA.com scraping
MLFLOW_TRACKING_URI  # (Optional) Remote MLflow server
```

## 📈 Roadmap

- [x] **Phase 0**: MLflow model registry integration
- [x] **Phase 0.5**: Feature schema validation and deterministic features
- [x] **Phase 1**: Inference module with artifact loading
- [x] **Phase 2**: Dashboard prep module
- [x] **Phase 3**: Streamlit dashboard with full features
- [x] **Phase 3.5**: Historical data and UI refinements
- [x] **Phase 3.6**: Prediction backfill and cache improvements
- [x] **Phase 4**: End-to-end pipeline orchestration
- [x] **Phase 5**: Docker containerization
- [ ] **Phase 6**: GitHub Actions deployment
- [ ] **Phase 7**: Streamlit Cloud deployment
- [ ] **Future**: Model retraining automation, A/B testing

## 🎓 Project Evolution

This is a comprehensive rework of [my original NBA prediction project](https://github.com/cmunch1/nba-prediction), focusing on:

**Reliability**: Open-source tools (MLflow, GitHub, Kaggle) instead of SaaS dependencies
**Engineering**: DI, interfaces, comprehensive testing, modular architecture
**Accuracy**: More features, better preprocessing, calibrated predictions
**Transparency**: Public datasets, detailed documentation, reproducible results

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- NBA.com for providing game statistics
- XGBoost, MLflow, Streamlit teams for excellent tools
- Kaggle for free dataset hosting
- GitHub Actions for CI/CD infrastructure

---

**Questions?** Open an issue or check the [documentation](docs/)

**Want to fork?** Run `./scripts/setup_fork.sh` to get started
