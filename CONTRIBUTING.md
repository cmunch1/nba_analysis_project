# Contributing to NBA Win Prediction System

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 Quick Start for Contributors

### Option 1: Use Public Kaggle Data (Recommended)

**No proxy or secrets required!** Perfect for testing, model improvements, or dashboard enhancements.

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/nba_analysis_project.git
cd nba_analysis_project

# 3. Run the setup wizard
./scripts/setup_fork.sh

# Or download data manually
./scripts/download_kaggle_data.sh

# 4. Run the ML pipeline
./scripts/run_nightly_pipeline.sh --data-source kaggle

# 5. Make your changes and test
# 6. Submit a pull request
```

### Option 2: Scrape Fresh Data (Advanced)

**Requires proxy service** for NBA.com scraping.

```bash
# Set up proxy (required)
export PROXY_URL="http://username:password@proxy-host:port"

# Run full pipeline with scraping
./scripts/run_nightly_pipeline.sh --data-source scrape
```

### Option 3: Use Docker

```bash
# Build and run with Docker Compose
docker-compose up nba-pipeline

# Or use pre-built image
docker run --rm \
  -v $(pwd)/data:/app/data \
  ghcr.io/YOUR_MAINTAINER_USERNAME/nba_analysis_project:latest
```

## 📋 Ways to Contribute

### 1. Report Issues

Found a bug? Have a suggestion? [Open an issue](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/issues/new)!

**Good issue template:**
```markdown
**Description**: Brief description of the issue

**Steps to Reproduce**:
1. Run command X
2. Observe error Y

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happened

**Environment**:
- OS: [e.g., Ubuntu 22.04]
- Docker: [Yes/No]
- Data Source: [Kaggle/Local/Scrape]
- Logs: [Attach relevant logs from logs/ directory]
```

### 2. Improve Documentation

Help make the project more accessible:
- Clarify confusing sections
- Add examples
- Fix typos
- Improve README or guides

**Small changes**: Edit directly on GitHub
**Larger changes**: Fork → edit → pull request

### 3. Enhance the Dashboard

The Streamlit dashboard is always improving! Ideas:
- New visualizations (team matchup history, player impact, etc.)
- Additional filters or metrics
- Performance optimizations
- Mobile responsiveness improvements

**Location**: `streamlit_app/`

**Test locally**:
```bash
streamlit run streamlit_app/app.py
```

### 4. Improve Models

Experiment with:
- **New features**: Add to `src/nba_app/feature_engineering/`
- **Different algorithms**: Add trainers to `src/ml_framework/model_testing/trainers/`
- **Hyperparameter tuning**: Update configs in `configs/core/hyperparameters/`
- **Better calibration**: Enhance `src/ml_framework/postprocessing/`

**Test models**:
```bash
# Train and evaluate
uv run -m src.ml_framework.model_testing.main

# Generate predictions
uv run -m src.nba_app.inference.main
```

### 5. Add Tests

Help improve code quality:
- Unit tests for new functions
- Integration tests for workflows
- Data validation tests

**Location**: `tests/`

**Run tests**:
```bash
uv run pytest tests/
```

### 6. Optimize Performance

- Speed up feature engineering
- Reduce Docker image size
- Optimize data loading
- Improve pipeline efficiency

**Benchmark before and after**:
```bash
time ./scripts/run_nightly_pipeline.sh --data-source kaggle
```

## 🔧 Development Setup

### Prerequisites

- Python 3.11+
- `uv` package manager (installed via setup script)
- Docker (optional but recommended)
- Git

### Environment Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/nba_analysis_project.git
cd nba_analysis_project

# 2. Install dependencies
pip install uv
uv sync

# 3. Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# 4. Download data
./scripts/download_kaggle_data.sh
```

### Project Structure

```
src/
├── nba_app/              # NBA-specific code
│   ├── webscraping/      # Data collection
│   ├── data_processing/  # Data cleaning
│   ├── feature_engineering/  # Feature creation
│   ├── inference/        # Prediction generation
│   └── dashboard_prep/   # Dashboard data prep
└── ml_framework/         # Reusable ML framework
    ├── core/             # Config, logging, DI
    ├── model_testing/    # Training & evaluation
    ├── preprocessing/    # Data preprocessing
    ├── postprocessing/   # Calibration & uncertainty
    └── model_registry/   # Model versioning
```

### Code Style

This project uses:
- **Dependency Injection**: All components injected via DI containers
- **Abstract Interfaces**: Base classes define contracts
- **Structured Logging**: Use `app_logger.structured_log()`
- **Type Hints**: Add type annotations to new code
- **Docstrings**: Document public methods

**See**: [docs/AI/core_framework_usage.md](docs/AI/core_framework_usage.md) for patterns

### Testing Your Changes

1. **Run pipeline end-to-end**:
```bash
./scripts/run_nightly_pipeline.sh --data-source kaggle
```

2. **Test specific module**:
```bash
uv run -m src.nba_app.feature_engineering.main
```

3. **Test with Docker**:
```bash
docker-compose up nba-pipeline
```

4. **Test dashboard**:
```bash
streamlit run streamlit_app/app.py
```

5. **Run unit tests** (if you added tests):
```bash
uv run pytest tests/ -v
```

## 📝 Pull Request Process

### Before Submitting

- [ ] Code runs successfully with Kaggle data
- [ ] No new errors in logs
- [ ] Dashboard still works (if applicable)
- [ ] Updated documentation (if applicable)
- [ ] Added tests (if applicable)
- [ ] Followed existing code patterns

### Submitting

1. **Create a branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

3. **Test thoroughly**

4. **Commit with clear messages**:
```bash
git add .
git commit -m "Add: Brief description of changes"
```

5. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

6. **Open pull request on GitHub**

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing Done
- [ ] Tested with Kaggle data
- [ ] Tested with Docker
- [ ] Dashboard still works
- [ ] No new errors in logs

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Additional Notes
Any other context or considerations
```

## 🎯 Good First Issues

New to the project? Look for issues labeled:
- `good first issue` - Easy tasks for beginners
- `documentation` - Documentation improvements
- `help wanted` - Community input requested

## 🤔 Questions?

- **General questions**: [Open a discussion](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/discussions)
- **Bug reports**: [Open an issue](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/issues)
- **Feature requests**: [Open an issue](https://github.com/YOUR_GITHUB_USERNAME/nba_analysis_project/issues) with "Feature Request" label

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behaviors:**
- Being respectful and inclusive
- Providing constructive feedback
- Accepting constructive criticism gracefully
- Focusing on what's best for the community

**Unacceptable behaviors:**
- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

### Enforcement

Project maintainers have the right to remove, edit, or reject contributions that don't align with this Code of Conduct.

## 🏆 Recognition

Contributors are recognized in:
- GitHub contributors list (automatic)
- Release notes for significant contributions
- Special thanks in README for major features

Thank you for contributing! 🎉

## 📚 Additional Resources

- [Core Framework Usage Guide](docs/AI/core_framework_usage.md)
- [Docker Deployment Guide](docs/DOCKER.md)
- [Streamlit Dashboard Reference](docs/streamlit_dashboard_reference.md)
- [Deployment Plan](DEPLOYMENT_PLAN.md)

---

**Not sure where to start?** Run `./scripts/setup_fork.sh` and explore the codebase!
