#!/bin/bash

# File: skysense_test.sh

# Description:
# Wrapper script to run the SkySense inference test.
# Installs dependencies and executes the Python test script.

# Notes: Output is captured and saved to a timestamped log file.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_DIR="/raid/inter_iit/isro/arnav/fine_tuning/logs"
LOG_FILE="$LOG_DIR/skysense_test_$(date +'%Y-%m-%d_%H-%M-%S').log"
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
    # Execution
    # -------------------------------------------------------------------------
    echo "[INFO] Starting SkySense Test Script..."
    python3 skysense_test.py

    echo "[INFO] SkySense testing finished successfully."
} 2>&1 | tee $LOG_FILE