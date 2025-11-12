# Dev Container vs Docker Compose: When to Use Each

This project supports both **VS Code Dev Containers** and **Docker Compose**. Here's how they work together.

## Quick Comparison

| Feature | Dev Container | Docker Compose |
|---------|--------------|----------------|
| **What it is** | Interactive development environment | Batch execution orchestration |
| **Use for** | Coding, debugging, testing | Running full pipeline |
| **VS Code** | Fully integrated | Run externally |
| **Terminal** | Inside container | Commands from host |
| **Debugging** | Full breakpoint support | Limited |
| **Extensions** | Work in container | Not applicable |
| **Jupyter** | Native support | Manual setup |
| **GPU** | Both configs available | Both compose files |
| **Startup** | VS Code command palette | `docker-compose up` |

## Use Cases

### Use Dev Container For:

**Daily Development Tasks:**
```python
# ✅ Write and edit code
# ✅ Set breakpoints and debug
# ✅ Run individual modules
python -m src.nba_app.feature_engineering.main

# ✅ Run specific tests
pytest tests/test_inference.py -v

# ✅ Interactive Python/IPython
python
>>> import torch
>>> torch.cuda.is_available()
True

# ✅ Jupyter notebooks
jupyter notebook
```

**What You Get:**
- IntelliSense and auto-complete work perfectly
- Python debugger with breakpoints
- Test explorer integration
- Git integration
- All VS Code extensions work
- Terminal is already inside container

### Use Docker Compose For:

**Pipeline Execution:**
```bash
# ✅ Run complete pipeline end-to-end
docker-compose -f docker-compose.gpu.yml up nba-pipeline-gpu

# ✅ Run specific pipeline stages
docker-compose run --rm nba-pipeline \
  python -m src.nba_app.webscraping.main

# ✅ Start the dashboard
docker-compose up nba-dashboard

# ✅ Simulate production
docker-compose -f docker-compose.gpu.yml up

# ✅ CI/CD testing locally
docker-compose run --rm nba-pipeline \
  ./scripts/run_nightly_pipeline.sh --data-source kaggle
```

**What You Get:**
- Full pipeline orchestration
- Multiple services (pipeline + dashboard)
- Exact production simulation
- Network between containers
- Volume management
- One command to run everything

## Typical Workflow

### Morning: Start Dev Container

```bash
# 1. Open VS Code
code .

# 2. Open Command Palette (Ctrl+Shift+P)
# 3. Type: "Dev Containers: Reopen in Container"
# 4. Select: "NBA Analysis Project (GPU)"

# Now you're inside the container!
# Terminal, editing, debugging all work
```

### Throughout the Day: Develop in Container

```python
# Edit code in VS Code
# File: src/nba_app/feature_engineering/main.py

# Set breakpoints in VS Code (click left of line number)
# Press F5 to debug

# Run tests
pytest tests/ -v

# Try things interactively
python -c "import torch; print(torch.cuda.is_available())"
```

### Evening: Test Full Pipeline

```bash
# Terminal 1: Keep dev container open (for quick edits)
# VS Code with dev container running

# Terminal 2: Run full pipeline to test
cd /home/chris/projects/nba_analysis_project
docker-compose -f docker-compose.gpu.yml up nba-pipeline-gpu
```

### Before Committing: Validate

```bash
# Inside dev container: Run tests
pytest tests/ -v

# Outside dev container: Full pipeline test
docker-compose -f docker-compose.gpu.yml run --rm nba-pipeline-gpu \
  ./scripts/run_nightly_pipeline.sh --data-source local

# If all good, commit from dev container
git add .
git commit -m "Your changes"
```

## Real-World Examples

### Example 1: Debugging a Feature Engineering Issue

**Problem**: Feature engineering is producing NaN values

**Solution using Dev Container:**
```python
# 1. Open in dev container (Ctrl+Shift+P → Reopen in Container)
# 2. Open: src/nba_app/feature_engineering/main.py
# 3. Set breakpoint at line where NaN might occur
# 4. Press F5 (Start Debugging)
# 5. Step through code, inspect variables
# 6. Find the issue, fix it
# 7. Run tests: pytest tests/test_feature_engineering.py
# 8. Commit fix
```

**Why Dev Container**: Immediate debugging with breakpoints, variable inspection

### Example 2: Testing End-to-End Pipeline

**Problem**: Need to verify full pipeline works before deploying

**Solution using Docker Compose:**
```bash
# Run complete pipeline exactly as it will run in GitHub Actions
docker-compose up nba-pipeline

# Check logs
tail -f logs/pipeline_*.log

# Verify predictions generated
ls -lh data/predictions/

# Test with GPU (your local setup)
docker-compose -f docker-compose.gpu.yml up nba-pipeline-gpu

# Compare timing
```

**Why Docker Compose**: Full orchestration, production simulation, easy logging

### Example 3: Developing a New Feature

**Problem**: Adding new rolling average features

**Solution using Both:**
```bash
# Step 1: Dev Container - Write code
# Open in dev container
# Edit: src/nba_app/feature_engineering/feature_creator.py
# Add new rolling average calculation
# Run: pytest tests/test_feature_creator.py -v -k rolling

# Step 2: Dev Container - Test interactively
python
>>> from src.nba_app.feature_engineering import FeatureCreator
>>> fc = FeatureCreator()
>>> # Test your new function

# Step 3: Docker Compose - Integration test
# Exit dev container (or open new terminal)
docker-compose run --rm nba-pipeline \
  python -m src.nba_app.feature_engineering.main

# Check output
head data/engineered/engineered_features.csv
```

**Why Both**: Dev container for quick iterations, compose for integration testing

## How They Share Resources

Both use the **same Docker images**:

```
Dockerfile (CPU) ──┬──> Dev Container (CPU)
                   └──> Docker Compose (CPU)

Dockerfile.gpu (GPU) ──┬──> Dev Container (GPU)
                       └──> Docker Compose (GPU)
```

**Benefits**:
- Build image once, use for both
- Consistent environment guaranteed
- Dev container = production environment

## Advanced: Using Both Simultaneously

You can have both running at the same time:

```bash
# Terminal 1: VS Code with Dev Container
# Doing interactive development, testing individual functions

# Terminal 2: Docker Compose
docker-compose up nba-dashboard
# Dashboard running at http://localhost:8501

# Terminal 3: Another compose service
docker-compose run --rm nba-pipeline \
  python -m src.nba_app.inference.main
```

**Use case**: Develop features in dev container, see results immediately in dashboard

## GPU Configuration

Both support GPU acceleration:

**Dev Container:**
```bash
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
# Choose: "NBA Analysis Project (GPU)"
# Uses devcontainer.gpu.json
```

**Docker Compose:**
```bash
docker-compose -f docker-compose.gpu.yml up
# Uses Dockerfile.gpu
```

**Same GPU, Same Performance**:
- Both use CUDA 12.1
- Both have GPU-enabled PyTorch/XGBoost
- Both access your RTX 3060
- Same 3-4x speedup

## Choosing Your Default Setup

### If you prefer interactive development:
1. Set dev container as default
2. Use it for 90% of work
3. Occasionally test with docker-compose

### If you prefer command-line:
1. Use docker-compose primarily
2. Occasionally open dev container for debugging
3. Use regular editor (vim/nano) for quick edits

### Hybrid approach (recommended):
1. **Dev container**: Writing new features, debugging
2. **Docker compose**: Testing full pipeline, running dashboard
3. **Direct Python**: Quick one-off scripts

## FAQ

**Q: Can I use both at the same time?**
A: Yes! They don't interfere with each other.

**Q: Which is faster to start?**
A: Dev container takes ~10 seconds, docker-compose up takes ~5 seconds (both after image is built).

**Q: Do they share the same built image?**
A: Yes, they can. Dev container can use the image built by docker-compose.

**Q: Can I debug with docker-compose?**
A: Limited. You can add `pdb` breakpoints, but no VS Code debugger integration.

**Q: Can I run the full pipeline in dev container?**
A: Yes! Just run `./scripts/run_nightly_pipeline.sh` in the dev container terminal.

**Q: Which uses more resources?**
A: Same. Both run the same container with the same resource limits.

## Summary

**Dev Container** = Your coding workspace (VS Code inside Docker)
**Docker Compose** = Your production simulator (running the pipeline)

Use **both** for the best experience:
- Develop interactively in dev container
- Test thoroughly with docker-compose
- Deploy with confidence knowing both use the same images

Your RTX 3060 works with both approaches! 🚀
