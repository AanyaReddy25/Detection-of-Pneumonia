#!/bin/bash
# ============================================================
# Download Chest X-Ray Pneumonia Dataset from Kaggle
# ============================================================
# Prerequisites:
#   1. Install Kaggle CLI:  pip install kaggle
#   2. Place your kaggle.json API key at ~/.kaggle/kaggle.json
#      (Download from https://www.kaggle.com/settings -> API -> Create New Token)
#
# Usage:
#   chmod +x download_dataset.sh
#   ./download_dataset.sh
# ============================================================

set -e

DATASET="paultimothymooney/chest-xray-pneumonia"
TARGET_DIR="code/data/input"

echo "============================================"
echo " Downloading Chest X-Ray Pneumonia Dataset"
echo "============================================"

# Check if kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "Error: 'kaggle' CLI not found."
    echo "Install it with: pip install kaggle"
    echo "Then place your API key at ~/.kaggle/kaggle.json"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Download dataset
echo "Downloading dataset from Kaggle..."
kaggle datasets download -d "$DATASET" -p "$TARGET_DIR" --unzip

echo ""
echo "============================================"
echo " Download complete!"
echo " Dataset extracted to: $TARGET_DIR/"
echo "============================================"
echo ""
echo "Expected structure:"
echo "  $TARGET_DIR/chest_xray/train/"
echo "  $TARGET_DIR/chest_xray/val/"
echo "  $TARGET_DIR/chest_xray/test/"
echo ""
echo "If the data is nested under chest_xray/, move it up:"
echo "  mv $TARGET_DIR/chest_xray/* $TARGET_DIR/"
echo "  rmdir $TARGET_DIR/chest_xray"
