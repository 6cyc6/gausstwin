#!/bin/bash

BASE_URL="https://huggingface.co/datasets/6cyc6/gausstwin_dataset/resolve/main"

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the dataset."
    exit 1
fi

echo "Downloading demo.tar.gz (7.07 GB)..."
$CMD "${BASE_URL}/demo.tar.gz" || { echo "Failed to download demo.tar.gz"; exit 1; }

echo "Extracting demo.tar.gz..."
tar -xzf demo.tar.gz && rm demo.tar.gz

echo "Done."
