# Dev Container Configuration

This project includes VS Code Dev Container configurations for interactive development inside Docker containers.

## What Are Dev Containers?

Dev Containers let you:
- ✅ Develop inside a container with VS Code
- ✅ Get consistent environment across machines
- ✅ Use VS Code extensions inside container
- ✅ Debug and run code interactively
- ✅ Access GPU from VS Code (with GPU config)

## Two Configurations Available

### 1. CPU Dev Container (Default)
- **File**: `devcontainer.json`
- **Use for**: General development, testing, quick edits
- **Works on**: Any machine
- **Size**: ~800 MB

### 2. GPU Dev Container
- **File**: `devcontainer.gpu.json`
- **Use for**: GPU-accelerated development, model training
- **Requires**: NVIDIA GPU + nvidia-docker
- **Size**: ~2.5 GB

## Quick Start

### Option A: Using Command Palette (Recommended)

1. **Open VS Code** in the project directory
2. **Press** `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. **Type**: "Dev Containers: Reopen in Container"
4. **Select**: CPU or GPU configuration

### Option B: Using the Prompt

When you open the project in VS Code, you'll see a prompt:
```
Folder contains a Dev Container configuration file.
Reopen folder to develop in a container?
```
Click **"Reopen in Container"**

### Switching Between CPU and GPU

**To use GPU version:**
1. Press `Ctrl+Shift+P`
2. Type: "Dev Containers: Reopen in Container"
3. Select "NBA Analysis Project (GPU)"

**To use CPU version:**
1. Press `Ctrl+Shift+P`
2. Type: "Dev Containers: Reopen in Container"
3. Select "NBA Analysis Project"

## What Gets Installed

When the dev container starts, it automatically installs:

**Extensions:**
- Python (with Pylance)
- Jupyter
- Ruff (Python linter)
- GitHub Copilot (if you have it)
- GitLens
- Docker extension
- YAML support

**Tools:**
- Zsh with Oh My Zsh
- Git
- All Python dependencies (from pyproject.toml)

## Development Workflow

### Inside the Dev Container

```bash
# Your terminal is already inside the container at /app

# Run tests
pytest tests/

# Run a single module
python -m src.nba_app.feature_engineering.main

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888

# Run the pipeline
./scripts/run_nightly_pipeline.sh --data-source local

# Check GPU (if using GPU container)
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### VS Code Features Work Normally

- **Debugging**: Set breakpoints, step through code
- **IntelliSense**: Auto-complete works with all imports
- **Testing**: Run tests from Test Explorer
- **Jupyter**: Create and run notebooks
- **Terminal**: Full bash/zsh terminal inside container

## Ports Automatically Forwarded

- **8501** - Streamlit dashboard
- **5000** - MLflow UI (if running)

Access them at `http://localhost:8501` and `http://localhost:5000`

## File Synchronization

All files are synchronized between your local machine and the container:
- ✅ Edit files in VS Code → changes appear in container
- ✅ Git operations work normally
- ✅ Commits are made with your local git config

## Comparison: Dev Container vs Docker Compose

| Feature | Dev Container | Docker Compose |
|---------|--------------|----------------|
| **Purpose** | Interactive development | Batch execution |
| **Use for** | Coding, debugging, testing | Running full pipeline |
| **VS Code** | Fully integrated | External |
| **Terminal** | Inside container | Run commands externally |
| **Extensions** | Work in container | Not applicable |
| **Debugging** | Full support | Limited |
| **Jupyter** | Native support | Manual setup |
| **Best for** | Day-to-day development | Production simulation |

## When to Use Each

**Use Dev Container when:**
- 🔧 Writing or debugging code
- 📊 Running Jupyter notebooks
- 🧪 Running individual tests
- 🔍 Exploring the codebase
- 💡 Experimenting with features

**Use Docker Compose when:**
- 🚀 Running the full pipeline end-to-end
- 📅 Simulating production/scheduled runs
- ⚡ Testing complete workflow
- 📦 Building for deployment
- 🎯 Running batch predictions

**Use both together:**
```bash
# Terminal 1: Dev Container in VS Code (for coding)
python -m src.nba_app.feature_engineering.main

# Terminal 2: Docker Compose (for full pipeline test)
docker-compose -f docker-compose.gpu.yml up nba-pipeline-gpu
```

## Troubleshooting

### Dev Container Won't Start

**Problem**: "Failed to start dev container"

**Solutions**:
1. Ensure Docker is running: `docker ps`
2. Build the base image first: `docker-compose build`
3. Check Docker logs: `docker logs <container-id>`

### GPU Not Available in Container

**Problem**: `torch.cuda.is_available()` returns False

**Solutions**:
1. Check host GPU: `nvidia-smi`
2. Verify nvidia-docker: `docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi`
3. Use GPU dev container config: `devcontainer.gpu.json`

### Extensions Not Working

**Problem**: Python extension shows errors

**Solutions**:
1. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"
2. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → `/app/.venv/bin/python`
3. Rebuild container: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"

### Git Operations Fail

**Problem**: Can't commit or push

**Solutions**:
1. Check git config is mounted: `git config --list`
2. Check SSH keys: `ls -la ~/.ssh`
3. Add to devcontainer.json mounts if needed

## Advanced Usage

### Custom Environment Variables

Edit `devcontainer.json` and add to `remoteEnv`:
```json
"remoteEnv": {
  "MLFLOW_TRACKING_URI": "http://your-mlflow-server.com",
  "CUSTOM_VAR": "value"
}
```

### Additional VS Code Settings

Edit `devcontainer.json` under `customizations.vscode.settings`:
```json
"python.linting.flake8Args": ["--max-line-length=120"]
```

### More Extensions

Add to `customizations.vscode.extensions`:
```json
"extensions": [
  "ms-python.python",
  "your-extension-id"
]
```

## Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Dev Container Specification](https://containers.dev/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## Summary

**Dev Container** = Your interactive development environment (VS Code inside Docker)
**Docker Compose** = Running the pipeline as a batch job

Both use the same Docker images, so your dev environment matches production exactly!
