#!/bin/bash

# File: evaluate.sh

# Description:
# Wrapper script to install evaluation dependencies,
# and execute the fine-tuning evaluation pipeline.

# Notes: Output is captured and saved to a timestamped log file.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/evaluate_$(date +'%Y-%m-%d_%H-%M-%S').log"
mkdir -p $LOG_DIR

# Wrap execution to capture stdout and stderr to the log file
{
    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------
    echo "[INFO] Installing dependencies for evaluation..."
    pip install -q -U transformers torch scikit-learn tqdm numpy

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    echo "[INFO] Starting Fine-tune Evaluation Script..."
    python3 evaluate.py

    echo "[INFO] Evaluation script finished successfully."
} 2>&1 | tee $LOG_FILE