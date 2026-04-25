#!/bin/bash

# File: download_vrsbench.sh

# Description:
# Automates the downloading and extraction of the VRSBench dataset 
# (Annotations, Images, and Evaluation JSONs) from HuggingFace.

# Notes: Requires 'wget' and 'unzip' to be installed.

# Exit immediately if any command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# Workspace Setup
# -----------------------------------------------------------------------------
echo "[INFO] Setting up dataset directory..."
mkdir -p vrsbench
cd vrsbench

# -----------------------------------------------------------------------------
# Source Configuration
# -----------------------------------------------------------------------------
# List of all required artifacts to download
files=(
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_train.zip"
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip"
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_train.zip"
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_Cap.json"
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_referring.json"
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_vqa.json"
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_train.json"
)

# -----------------------------------------------------------------------------
# Download Process
# -----------------------------------------------------------------------------
for url in "${files[@]}"; do
    echo "[INFO] Downloading: $url..."
    # -c allows continuing partially downloaded files
    wget -c "$url"
done

# -----------------------------------------------------------------------------
# Extraction Process
# -----------------------------------------------------------------------------
for z in *.zip; do
    echo "[INFO] Extracting archive: $z..."
    # -o overwrites existing files without prompting
    unzip -o "$z"
done

echo "[INFO] Dataset download and extraction completed successfully."