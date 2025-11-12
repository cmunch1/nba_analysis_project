# Docker Deployment Guide

This project provides Docker containers for consistent, reproducible execution across development and production environments.

## Quick Start

### Auto-Detect GPU Support

```bash
# Check if you have GPU available and get recommended setup
./scripts/detect_gpu.sh
```

### Local Development with Docker Compose

**CPU Version (Default - Works everywhere)**
```bash
# Build and run the entire pipeline
docker-compose up nba-pipeline

# Run with Kaggle data source (recommended)
docker-compose run --rm nba-pipeline /app/scripts/run_nightly_pipeline.sh --data-source kaggle

# Run Streamlit dashboard
docker-compose up nba-dashboard
# Access at http://localhost:8501
```

**GPU Version (For local development with NVIDIA GPU)**
```bash
# Check GPU availability first
./scripts/detect_gpu.sh

# Build and run with GPU support
docker-compose -f docker-compose.gpu.yml up nba-pipeline-gpu

# Run with Kaggle data source (GPU-accelerated)
docker-compose -f docker-compose.gpu.yml run --rm nba-pipeline-gpu \
  /app/scripts/run_nightly_pipeline.sh --data-source kaggle
```

### Manual Docker Commands

```bash
# Build the pipeline image
docker build -t nba-pipeline:latest .

# Build the Streamlit dashboard image
docker build -f Dockerfile.streamlit -t nba-dashboard:latest .

# Run pipeline with local data
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/mlruns:/app/mlruns \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  nba-pipeline:latest

# Run pipeline with Kaggle data
docker run --rm \
  -v $(pwd)/data:/app/data \
  nba-pipeline:latest \
  /app/scripts/run_nightly_pipeline.sh --data-source kaggle

# Run Streamlit dashboard
docker run --rm \
  -p 8501:8501 \
  -v $(pwd)/data/dashboard:/app/data/dashboard:ro \
  -v $(pwd)/data/predictions:/app/data/predictions:ro \
  nba-dashboard:latest
```

## Docker Images

### Pipeline Image (`Dockerfile`) - CPU Version

**Purpose**: Runs the complete ML pipeline (webscraping, processing, feature engineering, inference, dashboard prep)

**Base**: `python:3.11-slim`

**Includes**:
- Python 3.11
- All project dependencies (via uv)
- Chromium + ChromeDriver (for webscraping)
- Source code and configs
- CPU-optimized ML libraries

**Size**: ~800 MB (with multi-stage build optimization)

**Entry Point**: Configurable via CMD or docker-compose

**Use Cases**: GitHub Actions, CPU-only machines, forkers without GPU

### Pipeline Image (`Dockerfile.gpu`) - GPU Version

**Purpose**: Same as CPU version but with GPU acceleration for ML training/inference

**Base**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`

**Includes**:
- CUDA 12.1 + cuDNN 8
- Python 3.11
- GPU-enabled PyTorch, XGBoost, LightGBM, CatBoost
- Chromium + ChromeDriver (for webscraping)
- Source code and configs

**Size**: ~2.5 GB (includes CUDA runtime)

**Requirements**:
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed on host
- NVIDIA driver 525+ on host

**Entry Point**: Configurable via CMD or docker-compose

**Use Cases**: Local development with GPU, training large models, faster inference

### Streamlit Dashboard Image (`Dockerfile.streamlit`)

**Purpose**: Serves the interactive Streamlit dashboard

**Base**: `python:3.11-slim`

**Includes**:
- Python 3.11
- Streamlit and minimal dependencies
- Dashboard code only (no ML/scraping code)

**Size**: ~500 MB

**Exposes**: Port 8501

**Entry Point**: `streamlit run streamlit_app/app.py`

**Note**: CPU-only (no GPU needed for serving dashboard)

## Environment Variables

### Pipeline Container

```bash
# MLflow tracking (local or remote)
MLFLOW_TRACKING_URI=file:///app/mlruns
# or
MLFLOW_TRACKING_URI=https://your-mlflow-server.com

# Proxy for webscraping (optional)
PROXY_URL=http://username:password@proxy-host:port

# Python settings
PYTHONUNBUFFERED=1
```

### Streamlit Container

```bash
# Streamlit server settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

## Volume Mounts

### Required Volumes

```yaml
volumes:
  # Data persistence
  - ./data:/app/data

  # Logs (optional but recommended)
  - ./logs:/app/logs

  # MLflow artifacts (if using local tracking)
  - ./mlruns:/app/mlruns
```

### Read-Only Mounts

```yaml
volumes:
  # Configuration files (prevent accidental modification)
  - ./configs:/app/configs:ro

  # ML artifacts (models, feature allowlists)
  - ./ml_artifacts:/app/ml_artifacts:ro
```

## GitHub Actions Integration

Docker images are automatically built and pushed to GitHub Container Registry (ghcr.io) when:
- Code is pushed to `main` branch
- Dockerfile or dependencies change
- Weekly (Sundays at 1am EST) to pick up latest base image security updates

### Using Pre-Built Images

All GitHub Actions workflows use the pre-built Docker image:

```yaml
jobs:
  my_job:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/${{ github.repository }}:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
```

**Benefits:**
- ✅ Faster CI/CD (no dependency installation)
- ✅ Consistent environment across all workflows
- ✅ Easier debugging (can pull and run locally)

### Image Tags

| Tag | Description | Use Case |
|-----|-------------|----------|
| `latest` | Latest build from `main` branch | Production, scheduled jobs |
| `main-<sha>` | Specific commit from `main` | Reproducibility, debugging |
| `<branch>` | Latest build from branch | Testing feature branches |

## GPU Support

### Prerequisites

To use GPU acceleration locally, you need:

1. **NVIDIA GPU** with CUDA support (Compute Capability 3.5+)
2. **NVIDIA Driver** version 525 or newer
3. **nvidia-docker2** installed

### Setup nvidia-docker2

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Auto-Detection

Use the detection script to check your setup:

```bash
./scripts/detect_gpu.sh
```

This will:
- ✅ Detect if NVIDIA GPU is present
- ✅ Check if nvidia-docker is installed
- ✅ Show GPU specs (memory, driver version)
- ✅ Recommend appropriate docker-compose command

### Building GPU Image

```bash
# Build GPU-enabled image
docker build -f Dockerfile.gpu -t nba-pipeline:gpu .

# Or use docker-compose
docker-compose -f docker-compose.gpu.yml build
```

### Running with GPU

```bash
# Using docker-compose (recommended)
docker-compose -f docker-compose.gpu.yml up nba-pipeline-gpu

# Manual docker run
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  nba-pipeline:gpu
```

### Performance Comparison

Approximate speedups with GPU vs CPU (NVIDIA RTX 3090):

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| XGBoost Training | 45s | 12s | 3.8x |
| LightGBM Training | 38s | 15s | 2.5x |
| PyTorch Inference | 8s | 2s | 4.0x |
| Feature Engineering | 25s | 25s | 1.0x (not GPU-accelerated) |

**Note**: GitHub Actions does not provide GPU runners, so workflows always use CPU version.

### Choosing CPU vs GPU

**Use CPU version when:**
- Running on GitHub Actions
- No NVIDIA GPU available
- Small datasets (< 100k samples)
- Testing/development without GPU

**Use GPU version when:**
- Local development with NVIDIA GPU
- Training large models (> 100k samples)
- Hyperparameter tuning (many iterations)
- Want faster inference times

## Local Testing

### Test with Kaggle Data (No Secrets Required)

**CPU Version:**
```bash
# Option 1: Docker Compose
docker-compose run --rm nba-pipeline \
  /app/scripts/run_nightly_pipeline.sh --data-source kaggle

# Option 2: Manual Docker
docker run --rm \
  -v $(pwd)/data:/app/data \
  nba-pipeline:latest \
  bash -c "pip install kaggle && kaggle datasets download -d YOUR_USERNAME/nba-game-stats-daily -p data/cumulative_scraped --unzip && python -m src.nba_app.feature_engineering.main"
```

**GPU Version:**
```bash
# Check GPU first
./scripts/detect_gpu.sh

# Docker Compose with GPU
docker-compose -f docker-compose.gpu.yml run --rm nba-pipeline-gpu \
  /app/scripts/run_nightly_pipeline.sh --data-source kaggle

# Manual Docker with GPU
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  nba-pipeline:gpu \
  /app/scripts/run_nightly_pipeline.sh --data-source kaggle
```

### Test with Webscraping (Requires Proxy)

```bash
# Set proxy in .env file or export
export PROXY_URL="your-proxy-url"

docker-compose run --rm \
  -e PROXY_URL="$PROXY_URL" \
  nba-pipeline \
  /app/scripts/run_nightly_pipeline.sh --data-source scrape
```

### Interactive Shell for Debugging

```bash
# Enter container with bash
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs:ro \
  nba-pipeline:latest \
  bash

# Inside container:
# python -m src.nba_app.feature_engineering.main
# python -m src.nba_app.inference.main
```

## Production Deployment

### Option 1: GitHub Actions (Recommended)

See [.github/workflows/](.github/workflows/) for complete workflows.

**Pros:**
- Free compute (2,000 minutes/month)
- Automatic scheduling
- Built-in secrets management

**Cons:**
- 6-hour max runtime per job
- No persistent storage (use Kaggle or external storage)

### Option 2: Docker on VPS/Cloud

```bash
# Pull latest image
docker pull ghcr.io/YOUR_USERNAME/nba_analysis_project:latest

# Run with cron (example: 3am daily)
# Add to crontab:
0 3 * * * docker run --rm \
  -v /opt/nba-data:/app/data \
  -e MLFLOW_TRACKING_URI=https://your-mlflow.com \
  -e PROXY_URL=$PROXY_URL \
  ghcr.io/YOUR_USERNAME/nba_analysis_project:latest \
  /app/scripts/run_nightly_pipeline.sh
```

### Option 3: Kubernetes (Advanced)

```yaml
# Example CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nba-pipeline
spec:
  schedule: "0 8 * * *"  # 3am EST = 8am UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: nba-pipeline
            image: ghcr.io/YOUR_USERNAME/nba_analysis_project:latest
            env:
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                secretKeyRef:
                  name: mlflow-secret
                  key: tracking_uri
            volumeMounts:
            - name: data-volume
              mountPath: /app/data
          volumes:
          - name: data-volume
            persistentVolumeClaim:
              claimName: nba-data-pvc
          restartPolicy: OnFailure
```

## Troubleshooting

### Build Failures

**Issue**: Docker build fails with "out of space"

**Solution**: Prune old images and containers
```bash
docker system prune -a --volumes
```

**Issue**: Dependencies fail to install

**Solution**: Clear uv cache and rebuild
```bash
docker build --no-cache -t nba-pipeline:latest .
```

### Runtime Issues

**Issue**: "Permission denied" errors in container

**Solution**: Check volume mount permissions
```bash
# Fix permissions on host
chmod -R 755 data/ logs/ mlruns/
```

**Issue**: Kaggle datasets not downloading

**Solution**: For public datasets, no authentication needed. If still failing:
```bash
# Test download manually
docker run --rm nba-pipeline:latest \
  bash -c "pip install kaggle && kaggle datasets download -d YOUR_USERNAME/dataset-name"
```

**Issue**: MLflow artifacts not loading

**Solution**: Ensure MLflow tracking URI is correct
```bash
# Check MLflow is accessible
docker run --rm nba-pipeline:latest \
  bash -c "python -c 'import mlflow; print(mlflow.get_tracking_uri())'"
```

### Dashboard Issues

**Issue**: Dashboard shows "No data available"

**Solution**: Ensure dashboard data volume is mounted correctly
```bash
# Check volume mount
docker-compose run --rm nba-dashboard ls -la /app/data/dashboard
```

**Issue**: Can't access dashboard on port 8501

**Solution**: Check port mapping and firewall
```bash
# Test if Streamlit is running
curl http://localhost:8501/_stcore/health
```

## Best Practices

### Development
- Use `docker-compose` for local testing
- Mount volumes for live code reloading (add to docker-compose.yml)
- Use `.env` file for environment variables

### Production
- Use pre-built images from GHCR (don't build in production)
- Set resource limits (memory, CPU)
- Configure proper logging and monitoring
- Use health checks for container orchestration

### Security
- Never commit secrets to Dockerfile
- Use Docker secrets or environment variables
- Run containers as non-root user (already configured)
- Keep base images updated (weekly rebuild via GitHub Actions)

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [GitHub Actions with Docker](https://docs.github.com/en/actions/publishing-packages/publishing-docker-images)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
