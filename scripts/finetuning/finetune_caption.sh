#!/bin/bash

# File: finetune_caption.sh

# Description:
# Orchestrates the Qwen-VL fine-tuning pipeline for captioning.
# Installs dependencies, runs the training script,
# and immediately triggers the inference script for validation.

# Notes: Output is logged to a timestamped file in the logs directory.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/finetune_caption_$(date +'%Y-%m-%d_%H-%M-%S').log"
mkdir -p $LOG_DIR

# Wrap execution to capture stdout and stderr to the log file
{
    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------
    echo "[INFO] Installing necessary dependencies..."
    pip install -q -U \
        torch torchvision Pillow transformers accelerate bitsandbytes \
        qwen-vl-utils peft scikit-learn tensorboard

    # -------------------------------------------------------------------------
    # Training Phase
    # -------------------------------------------------------------------------
    echo "[INFO] Starting Qwen-VL Fine-Tuning Script..."
    python3 finetune_caption.py

    # -------------------------------------------------------------------------
    # Inference Phase
    # -------------------------------------------------------------------------
    echo "[INFO] Fine-tuning completed. Starting Inference Script..."
    python3 inference_caption.py

    echo "[INFO] Pipeline finished successfully."
} 2>&1 | tee $LOG_FILE