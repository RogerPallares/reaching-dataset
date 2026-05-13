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

# Figshare article API endpoint
ARTICLE_API_URL="https://api.figshare.com/v2/articles/31030252"
TEMP_DIR=$(mktemp -d)
DATA_ZIP="$TEMP_DIR/data.zip"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    echo "Done!"
}

# Register cleanup function to run on exit
trap cleanup EXIT

# Check if curl or wget is available
if command -v curl &> /dev/null; then
    DOWNLOAD_TOOL="curl"
elif command -v wget &> /dev/null; then
    DOWNLOAD_TOOL="wget"
else
    echo "Error: Neither wget nor curl is installed. Please install one of them."
    exit 1
fi

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python is required to parse Figshare API metadata."
    exit 1
fi

download_file() {
    local url="$1"
    local output="$2"

    if [ "$DOWNLOAD_TOOL" = "curl" ]; then
        curl -L --fail -o "$output" "$url"
    else
        wget -O "$output" "$url"
    fi
}

# Fetch public file metadata from the Figshare API
echo "Fetching file metadata from figshare..."
METADATA_JSON="$TEMP_DIR/article.json"
download_file "$ARTICLE_API_URL" "$METADATA_JSON"

FILES_TSV="$TEMP_DIR/files.tsv"
"$PYTHON_CMD" - "$METADATA_JSON" > "$FILES_TSV" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    article = json.load(f)

files = article.get("files", [])
if not files:
    raise SystemExit("No downloadable files were found in the Figshare article metadata.")

for file_info in files:
    print(
        file_info["name"],
        file_info["download_url"],
        file_info["size"],
        file_info.get("computed_md5") or "",
        sep="\t",
    )
PY

# Download each file listed by the API
echo "Downloading dataset files from figshare..."
echo "This may take several minutes depending on your connection speed."
echo ""

while IFS=$'\t' read -r name url expected_size expected_md5; do
    output="$TEMP_DIR/$name"
    size_mb=$("$PYTHON_CMD" - "$expected_size" <<'PY'
import sys
print(round(int(sys.argv[1]) / 1024 / 1024, 2))
PY
)
    echo "  Downloading $name (${size_mb} MB)..."
    download_file "$url" "$output"

    actual_size=$(wc -c < "$output" | tr -d ' ')
    if [ "$actual_size" != "$expected_size" ]; then
        echo "Error: $name has size $actual_size bytes, expected $expected_size bytes."
        exit 1
    fi

    if [ -n "$expected_md5" ]; then
        if command -v md5sum &> /dev/null; then
            actual_md5=$(md5sum "$output" | awk '{print $1}')
        elif command -v md5 &> /dev/null; then
            actual_md5=$(md5 -q "$output")
        else
            echo "Warning: Neither md5sum nor md5 is installed; skipping checksum for $name."
            actual_md5="$expected_md5"
        fi

        if [ "$actual_md5" != "$expected_md5" ]; then
            echo "Error: $name MD5 checksum mismatch."
            exit 1
        fi
    fi
done < "$FILES_TSV"

echo ""
echo "Downloads complete!"
echo ""

# Extract data.zip
if [ -f "$DATA_ZIP" ]; then
    echo "Extracting data.zip to destination folder..."
    "$PYTHON_CMD" - "$DATA_ZIP" "$DEST_PATH" <<'PY'
import sys
import zipfile

zip_path, dest_path = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(dest_path)
PY
else
    echo "Warning: data.zip not found in the archive."
fi

# Copy other files (dataset.csv, exceptions.txt, etc.) to destination
echo "Copying metadata files to destination folder..."

FILES_TO_COPY=("README.txt" "dataset.csv" "SHA256SUMS.txt")

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
