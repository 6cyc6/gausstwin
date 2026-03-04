#!/bin/bash

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"

# SAM 2.1 checkpoints
echo "Downloading sam2.1_hiera_small.pt checkpoint..."
$CMD $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url"; exit 1; }

# EfficientTAM checkpoint
EFFICIENTTAM_URL="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt"
echo "Downloading efficienttam_s_512x512.pt checkpoint..."
$CMD $EFFICIENTTAM_URL || { echo "Failed to download checkpoint from $EFFICIENTTAM_URL"; exit 1; }

echo "All checkpoints are downloaded successfully."