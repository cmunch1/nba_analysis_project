#!/bin/bash
# GPU Detection Script
# Auto-detects GPU availability and recommends appropriate Docker setup

set -e

echo "=== GPU Detection ==="
echo ""

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""

    # Check if nvidia-docker is installed
    if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo "✓ nvidia-docker runtime is available"
        echo ""
        echo "Recommended setup:"
        echo "  docker-compose -f docker-compose.gpu.yml up"
        echo ""
        echo "Or build GPU image:"
        echo "  docker build -f Dockerfile.gpu -t nba-pipeline:gpu ."
        exit 0
    else
        echo "⚠ nvidia-docker runtime not detected"
        echo ""
        echo "To enable GPU support in Docker:"
        echo "  1. Install nvidia-docker2:"
        echo "     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
        echo "     distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "     curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo "     sudo apt-get update && sudo apt-get install -y nvidia-docker2"
        echo "     sudo systemctl restart docker"
        echo ""
        echo "  2. Test GPU access:"
        echo "     docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
        echo ""
        echo "For now, use CPU version:"
        echo "  docker-compose up"
        exit 1
    fi
else
    echo "✗ No NVIDIA GPU detected"
    echo ""
    echo "Running on CPU. Use standard Docker setup:"
    echo "  docker-compose up"
    echo ""
    echo "If you have a GPU but nvidia-smi is not available:"
    echo "  sudo apt-get install nvidia-utils-XXX  (replace XXX with your driver version)"
    exit 2
fi
