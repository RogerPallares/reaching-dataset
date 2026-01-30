
<#
.SYNOPSIS
    Download and extract the reaching dataset from figshare.

.DESCRIPTION
    This script downloads the reaching dataset from figshare, extracts it,
    and organizes the files in the specified destination folder.

.PARAMETER DestPath
    The destination path where the dataset will be extracted.

.EXAMPLE
    .\download_dataset.ps1 -DestPath "C:\path\to\dataset"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DestPath
)

# Create destination directory if it doesn't exist
if (-not (Test-Path $DestPath)) {
    Write-Host "Creating destination directory: $DestPath"
    New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
}

# Convert to absolute path
$DestPath = (Resolve-Path $DestPath).Path

Write-Host "Destination path: $DestPath"
Write-Host ""

# Download URL
$downloadUrl = "https://figshare.com/ndownloader/articles/31030252"
$tempDir = Join-Path $env:TEMP "reaching_dataset_temp"
$zipFile = Join-Path $tempDir "31030252.zip"

# Create temporary directory
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}

try {
    # Download the dataset
    Write-Host "Downloading dataset from figshare..."
    Write-Host "This may take several minutes depending on your connection speed."
    Write-Host ""
    
    # Use Invoke-WebRequest with progress
    $ProgressPreference = 'Continue'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
    
    Write-Host "Download complete!"
    Write-Host ""
    
    # Extract the main archive
    Write-Host "Extracting main archive..."
    Expand-Archive -Path $zipFile -DestinationPath $tempDir -Force
    
    # Find and extract data.zip
    $dataZipPath = Join-Path $tempDir "data.zip"
    
    if (Test-Path $dataZipPath) {
        Write-Host "Extracting data.zip to destination folder..."
        Expand-Archive -Path $dataZipPath -DestinationPath $DestPath -Force
    } else {
        Write-Host "Warning: data.zip not found in the archive."
    }
    
    # Copy other files (dataset.csv, exceptions.txt, etc.) to destination
    Write-Host "Copying metadata files to destination folder..."
    $filesToCopy = @("dataset.csv", "exceptions.txt", "hdf5_structure.txt", "SHA256SUMS.txt")
    
    foreach ($file in $filesToCopy) {
        $sourcePath = Join-Path $tempDir $file
        if (Test-Path $sourcePath) {
            Copy-Item -Path $sourcePath -Destination $DestPath -Force
            Write-Host "  Copied: $file"
        }
    }
    
    Write-Host ""
    Write-Host "Dataset successfully downloaded and extracted to: $DestPath"
    Write-Host ""
    Write-Host "The dataset folder should now contain:"
    Write-Host "  - dataset.csv"
    Write-Host "  - hdf5_files/"
    Write-Host "  - us_videos/"
    Write-Host "  - exceptions.txt"
    Write-Host "  - hdf5_structure.txt"
    
} catch {
    Write-Host "Error occurred: $_" -ForegroundColor Red
    exit 1
} finally {
    # Cleanup temporary directory
    Write-Host ""
    Write-Host "Cleaning up temporary files..."
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
    Write-Host "Done!"
}
