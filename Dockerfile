# NBA Analysis Project - Docker Image
# Multi-stage build for optimized image size

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /build

# Copy dependency files (README.md required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies into /build/.venv
RUN uv sync --frozen

# Stage 2: Runtime - Minimal image with only what's needed
FROM python:3.11-slim

# Install runtime dependencies (bash included in slim image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install uv for runtime package management
RUN pip install --no-cache-dir uv

# Create non-root user for security
RUN useradd -m -u 1000 nba && \
    mkdir -p /app && \
    chown -R nba:nba /app

# Set working directory
WORKDIR /app

# Copy Python virtual environment from builder
COPY --from=builder --chown=nba:nba /build/.venv /app/.venv

# Copy application code
COPY --chown=nba:nba src/ ./src/
COPY --chown=nba:nba configs/ ./configs/
COPY --chown=nba:nba scripts/ ./scripts/
COPY --chown=nba:nba pyproject.toml README.md ./

# Ensure scripts are executable
RUN chmod +x scripts/*.sh scripts/*.py

# Create data directories
RUN mkdir -p \
    data/cumulative_scraped \
    data/newly_scraped \
    data/processed \
    data/engineered \
    data/predictions \
    data/dashboard \
    logs \
    mlruns \
    && chown -R nba:nba data logs mlruns

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Install Kaggle CLI for Kaggle workflow support (before switching to nba user)
RUN uv pip install --system kaggle

# Switch to non-root user
USER nba

# Environment variables (override at runtime)
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"
ENV PYTHONUNBUFFERED=1

# Default command runs the full pipeline
CMD ["python", "-m", "src.nba_app.inference.main"]

# Health check (for container orchestration)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"
