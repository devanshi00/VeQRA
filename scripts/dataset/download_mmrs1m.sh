#!/bin/bash

# File: download_mmrs1m.sh

# Description:
# Automates the download and extraction of specific SAR and IR 
# detection subsets from the MMRS-1M split archive.

# Notes:
# The dataset is hosted as multi-part gzip files. We reconstruct
# the stream on-the-fly to extract only relevant directories.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
REPO_URL="https://huggingface.co/datasets/initiacms/MMRS-1M/resolve/main"
TARGET_DIR="mmrs-1m"
TEMP_DIR="mmrs_temp"

# Ensure target directory exists; use temp directory for download artifacts
mkdir -p "$TARGET_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# -----------------------------------------------------------------------------
# Download Process
# -----------------------------------------------------------------------------
echo "[INFO] Starting download of MMRS-1M split archive parts (0-6)..."

# List of split files required to reconstruct the dataset
parts=(
    "data.tar.gz.0"
    "data.tar.gz.1"
    "data.tar.gz.2"
    "data.tar.gz.3"
    "data.tar.gz.4"
    "data.tar.gz.5"
    "data.tar.gz.6"
)

for part in "${parts[@]}"; do
    if [ ! -f "$part" ]; then
        echo "[INFO] Downloading: $part..."
        # -c continues download, -q --show-progress keeps output clean but visible
        wget -c -q --show-progress "$REPO_URL/$part"
    else
        echo "[INFO] $part already exists. Skipping."
    fi
done

# -----------------------------------------------------------------------------
# Extraction Process
# -----------------------------------------------------------------------------
echo "[INFO] Extracting verified SAR and IR datasets from archive stream..."

# 1. Extract raw image/data folders
# We cat all parts together and pipe directly to tar to avoid creating a massive intermediate file.
# Wildcards are used to selectively extract only specific SAR/IR datasets.
cat data.tar.gz.* | tar -xzf - -C "../$TARGET_DIR" \
    --wildcards \
    'data/detection/HRSID*' \
    'data/detection/SARV2*' \
    'data/detection/IR_*' \
    'data/detection/infrared_ship_*'

# 2. Extract corresponding JSON annotations
echo "[INFO] Extracting detection JSON annotations..."
cat data.tar.gz.* | tar -xzf - -C "../$TARGET_DIR" \
    --wildcards \
    'data/json/detection/*'

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo "[INFO] Cleaning up temporary download files..."
cd ..
rm -rf "$TEMP_DIR"

echo "[INFO] Extraction complete. Dataset located in: $TARGET_DIR"