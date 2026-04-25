#!/bin/bash

# File: run.sh

# Description:
# Orchestrates the full benchmarking pipeline: dependency
# installation, model inference, and metric evaluation.

# Notes: Output is logged to a timestamped file in the logs directory.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/benchmark_$(date +'%Y-%m-%d_%H-%M-%S').log"
mkdir -p $LOG_DIR

# Wrap execution in a block to capture all stdout/stderr to the log file
{
    # -------------------------------------------------------------------------
    # Dependency Installation
    # -------------------------------------------------------------------------
    echo "[INFO] Installing system dependencies..."
    pip install -q -U \
        torch torchvision transformers accelerate \
        bitsandbytes qwen-vl-utils scikit-learn peft

    # -------------------------------------------------------------------------
    # Model Inference
    # -------------------------------------------------------------------------
    echo "[INFO] Starting Qwen-VL inference script..."
    python3 qwenvl.py

    echo "[INFO] Starting LLaVA-NeXT inference script..."
    python3 llavanext.py

    echo "[INFO] Starting RS-LLaVA inference script..."
    python3 rsllava.py

    echo "[INFO] All inference tasks completed."

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    echo "[INFO] Starting evaluation metrics script..."
    python3 evaluate.py

    echo "[INFO] Benchmarking pipeline finished successfully."
} 2>&1 | tee $LOG_FILE