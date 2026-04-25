#!/bin/bash

# File: finetune_vqa.sh

# Description:
# Orchestrates the VQA-specific fine-tuning pipeline.
# Installs dependencies, trains the model on VQA data,
# and immediately runs inference validation.

# Notes: Output is captured and saved to a timestamped log file.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/finetune_vqa_$(date +'%Y-%m-%d_%H-%M-%S').log"
mkdir -p $LOG_DIR

# Wrap execution to capture stdout and stderr to the log file
{
    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------
    echo "[INFO] Checking and installing dependencies..."
    pip install -q -U \
        torch torchvision Pillow transformers accelerate bitsandbytes \
        qwen-vl-utils peft scikit-learn tensorboard pandas

    # -------------------------------------------------------------------------
    # Training Phase
    # -------------------------------------------------------------------------
    echo "[INFO] Starting VQA Fine-Tuning Script..."
    python3 finetune_vqa.py

    # -------------------------------------------------------------------------
    # Inference Phase
    # -------------------------------------------------------------------------
    echo "[INFO] Training finished. Starting VQA Inference Script..."
    python3 inference_vqa.py

    echo "[INFO] VQA pipeline finished successfully."
} 2>&1 | tee $LOG_FILE