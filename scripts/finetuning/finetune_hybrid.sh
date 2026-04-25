#!/bin/bash

# File: finetune_hybrid.sh

# Description:
# Orchestrates the Hybrid Fine-Tuning pipeline (VRSBench + SkySense).
# Installs dependencies, trains the model,
# and immediately runs inference validation.

# Notes: Output is captured and saved to a timestamped log file.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/finetune_hybrid_$(date +'%Y-%m-%d_%H-%M-%S').log"
mkdir -p $LOG_DIR

# Wrap execution to capture stdout and stderr to the log file
{
    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------
    echo "[INFO] Installing dependencies (including pandas for SkySense CSVs)..."
    pip install -q -U \
        torch torchvision Pillow transformers accelerate bitsandbytes \
        qwen-vl-utils peft scikit-learn tensorboard pandas

    # -------------------------------------------------------------------------
    # Training Phase
    # -------------------------------------------------------------------------
    echo "[INFO] Starting Hybrid Fine-Tuning Script..."
    python3 finetune_hybrid.py

    # -------------------------------------------------------------------------
    # Inference Phase
    # -------------------------------------------------------------------------
    echo "[INFO] Training finished. Starting Hybrid Inference Script..."
    python3 inference_hybrid.py

    echo "[INFO] Hybrid pipeline finished successfully."
} 2>&1 | tee $LOG_FILE