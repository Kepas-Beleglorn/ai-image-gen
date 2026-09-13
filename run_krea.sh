#!/bin/bash

# ------------------------------------------------------------------
# Krea 2 Turbo Image Generator - Low VRAM Mode
# Optimized for RTX 3060 (12GB) with CPU Offloading
# ------------------------------------------------------------------

# 1. Define Paths
PROJECT_DIR="/mnt/data/pycharm_projects/ai-image-gen"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
APP_SCRIPT="$PROJECT_DIR/app.py"

# 2. Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PYTHON"
    echo "Please ensure you have installed the dependencies in PyCharm."
    exit 1
fi

echo "------------------------------------------------------------"
echo "Starting Krea 2 Turbo Generator..."
echo "Project: $PROJECT_DIR"
echo "Python:  $VENV_PYTHON"
echo "Mode:    Low VRAM + Disable Smart Memory"
echo "GPU:     RTX 3060 (Device 1)"
echo "------------------------------------------------------------"

# 3. Set Environment Variables
# Force usage of the secondary GPU (RTX 3060)
export CUDA_VISIBLE_DEVICES=1

# Prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable NCCL (fixes the 'undefined symbol' error for single-user inference)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Force ComfyUI Low VRAM mode explicitly
export COMFYUI_LOWVRAM=true

# 4. Run the Application
# We call the python binary directly from the venv (equivalent to activating it)
"$VENV_PYTHON" "$APP_SCRIPT" --lowvram --disable-smart-memory

# 5. Handle Exit
if [ $? -ne 0 ]; then
    echo "------------------------------------------------------------"
    echo "Application exited with an error."
    echo "Check the logs above for 'CUDA out of memory' or other issues."
    echo "------------------------------------------------------------"
else
    echo "------------------------------------------------------------"
    echo "Application stopped gracefully."
    echo "------------------------------------------------------------"
fi

