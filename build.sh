#!/usr/bin/env bash
# Render build script:
# 1. Install Python dependencies
# 2. Download the AI model weights if not present

set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Checking AI model ==="
mkdir -p model

if [ ! -f "model/grape_disease_model.pth" ]; then
    echo "Downloading grape disease model from GitHub Releases..."
    curl -L -o model/grape_disease_model.pth \
        "https://github.com/adityagolande2406-hub/agrisense-backend/releases/download/v1.0.0/grape_disease_model.pth"
    echo "Model downloaded: $(du -sh model/grape_disease_model.pth)"
else
    echo "Model already present: $(du -sh model/grape_disease_model.pth)"
fi

echo "=== Build complete ==="
