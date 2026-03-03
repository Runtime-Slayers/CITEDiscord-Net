#!/usr/bin/env bash
# =============================================================================
# setup.sh — CITEDiscord-Net Environment Setup
# =============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== CITEDiscord-Net Setup ==="
echo "Project directory: $PROJ_DIR"

# Python version check
PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PY_VER"

# Create virtual environment if not exists
VENV_DIR="$PROJ_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip --quiet

# Install PyTorch (CPU or CUDA auto-selection)
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA detected — installing GPU PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "No CUDA — installing CPU PyTorch"
    pip install torch torchvision torchaudio --quiet
fi

# Install project requirements
pip install -r "$PROJ_DIR/requirements.txt" --quiet

# Create data directories
mkdir -p "$PROJ_DIR/data/raw"
mkdir -p "$PROJ_DIR/data/processed"
mkdir -p "$PROJ_DIR/data/augmented"
mkdir -p "$PROJ_DIR/results"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with:      python main.py"
