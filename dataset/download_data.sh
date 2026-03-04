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

declare -A FILES=(
    ["multi.tar.gz"]="14.1 GB"
    ["pushover.tar.gz"]="9.59 GB"
    ["rope.tar.gz"]="14.3 GB"
    ["single.tar.gz"]="18.3 GB"
)

for FILE in "${!FILES[@]}"; do
    SIZE="${FILES[$FILE]}"
    echo "Downloading ${FILE} (${SIZE})..."
    $CMD "${BASE_URL}/${FILE}" || { echo "Failed to download ${FILE}"; exit 1; }

    echo "Extracting ${FILE}..."
    tar -xzf "${FILE}" && rm "${FILE}"
done

echo "All files downloaded and extracted successfully."
