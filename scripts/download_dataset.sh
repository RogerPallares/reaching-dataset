#!/bin/bash

# Download and extract the reaching dataset from figshare
#
# Usage: ./download_dataset.sh /path/to/destination

set -e  # Exit on error

# Check if destination path is provided
if [ $# -eq 0 ]; then
    echo "Error: No destination path provided"
    echo "Usage: $0 /path/to/destination"
    exit 1
fi

DEST_PATH="$1"

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_PATH" ]; then
    echo "Creating destination directory: $DEST_PATH"
    mkdir -p "$DEST_PATH"
fi

# Convert to absolute path
DEST_PATH=$(cd "$DEST_PATH" && pwd)

echo "Destination path: $DEST_PATH"
echo ""

# Download URL
DOWNLOAD_URL="https://figshare.com/ndownloader/articles/31030252"
TEMP_DIR=$(mktemp -d)
ZIP_FILE="$TEMP_DIR/31030252.zip"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    echo "Done!"
}

# Register cleanup function to run on exit
trap cleanup EXIT

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
else
    echo "Error: Neither wget nor curl is installed. Please install one of them."
    exit 1
fi

# Download the dataset
echo "Downloading dataset from figshare..."
echo "This may take several minutes depending on your connection speed."
echo ""

$DOWNLOAD_CMD "$ZIP_FILE" "$DOWNLOAD_URL"

echo ""
echo "Download complete!"
echo ""

# Extract the main archive
echo "Extracting main archive..."
unzip -q "$ZIP_FILE" -d "$TEMP_DIR"

# Find and extract data.zip
DATA_ZIP="$TEMP_DIR/data.zip"

if [ -f "$DATA_ZIP" ]; then
    echo "Extracting data.zip to destination folder..."
    unzip -q "$DATA_ZIP" -d "$DEST_PATH"
else
    echo "Warning: data.zip not found in the archive."
fi

# Copy other files (dataset.csv, exceptions.txt, etc.) to destination
echo "Copying metadata files to destination folder..."

FILES_TO_COPY=("dataset.csv" "exceptions.txt" "hdf5_structure.txt" "SHA256SUMS.txt")

for file in "${FILES_TO_COPY[@]}"; do
    SOURCE_PATH="$TEMP_DIR/$file"
    if [ -f "$SOURCE_PATH" ]; then
        cp "$SOURCE_PATH" "$DEST_PATH/"
        echo "  Copied: $file"
    fi
done

echo ""
echo "Dataset successfully downloaded and extracted to: $DEST_PATH"
echo ""
echo "The dataset folder should now contain:"
echo "  - dataset.csv"
echo "  - hdf5_files/"
echo "  - us_videos/"
echo "  - exceptions.txt"
echo "  - hdf5_structure.txt"
