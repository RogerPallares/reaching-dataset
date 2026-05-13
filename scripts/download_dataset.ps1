
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

function Download-File {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Url,
        [Parameter(Mandatory=$true)]
        [string]$OutFile,
        [Parameter(Mandatory=$true)]
        [Int64]$ExpectedSize,
        [Parameter(Mandatory=$true)]
        [string]$Name
    )

    Add-Type -AssemblyName System.Net.Http

    $client = [System.Net.Http.HttpClient]::new()
    $response = $null
    $inputStream = $null
    $outputStream = $null

    try {
        $response = $client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $response.EnsureSuccessStatusCode() | Out-Null

        $totalBytes = $response.Content.Headers.ContentLength
        if (-not $totalBytes) {
            $totalBytes = $ExpectedSize
        }

        $inputStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $outputStream = [System.IO.File]::Create($OutFile)
        $buffer = New-Object byte[] (1024 * 1024)
        $downloadedBytes = [Int64]0
        $lastProgress = Get-Date

        while (($bytesRead = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $outputStream.Write($buffer, 0, $bytesRead)
            $downloadedBytes += $bytesRead

            $now = Get-Date
            if (($now - $lastProgress).TotalMilliseconds -ge 500 -or $downloadedBytes -eq $totalBytes) {
                $percent = [math]::Min(100, [math]::Round(($downloadedBytes / $totalBytes) * 100, 1))
                $status = "$([math]::Round($downloadedBytes / 1MB, 1)) MB / $([math]::Round($totalBytes / 1MB, 1)) MB"
                Write-Progress -Activity "Downloading $Name" -Status $status -PercentComplete $percent
                $lastProgress = $now
            }
        }

        Write-Progress -Activity "Downloading $Name" -Completed
    } finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        if ($response) { $response.Dispose() }
        if ($client) { $client.Dispose() }
    }
}

# Create destination directory if it doesn't exist
if (-not (Test-Path $DestPath)) {
    Write-Host "Creating destination directory: $DestPath"
    New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
}

# Convert to absolute path
$DestPath = (Resolve-Path $DestPath).Path

Write-Host "Destination path: $DestPath"
Write-Host ""

# Figshare article API endpoint
$articleApiUrl = "https://api.figshare.com/v2/articles/31030252"
$tempDir = Join-Path $env:TEMP ("reaching_dataset_" + [guid]::NewGuid().ToString("N"))
$dataZipPath = Join-Path $tempDir "data.zip"

# Create temporary directory
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    # Fetch public file metadata from the Figshare API
    Write-Host "Fetching file metadata from figshare..."
    $article = Invoke-RestMethod -Uri $articleApiUrl -Method Get
    $files = $article.files

    if (-not $files -or $files.Count -eq 0) {
        throw "No downloadable files were found for article $articleApiUrl."
    }

    # Download each file listed by the API
    Write-Host "Downloading dataset files from figshare..."
    Write-Host "This may take several minutes depending on your connection speed."
    Write-Host ""

    foreach ($file in $files) {
        $outPath = Join-Path $tempDir $file.name
        Write-Host "  Downloading $($file.name) ($([math]::Round($file.size / 1MB, 2)) MB)..."
        Download-File -Url $file.download_url -OutFile $outPath -ExpectedSize $file.size -Name $file.name

        if (-not (Test-Path $outPath)) {
            throw "Download failed: $($file.name) was not created."
        }

        $downloadedSize = (Get-Item $outPath).Length
        if ($downloadedSize -ne [int64]$file.size) {
            throw "Download failed: $($file.name) has size $downloadedSize bytes, expected $($file.size) bytes."
        }

        if ($file.computed_md5) {
            $downloadedMd5 = (Get-FileHash -Path $outPath -Algorithm MD5).Hash.ToLowerInvariant()
            if ($downloadedMd5 -ne $file.computed_md5.ToLowerInvariant()) {
                throw "Download failed: $($file.name) MD5 checksum mismatch."
            }
        }
    }

    Write-Host ""
    Write-Host "Downloads complete!"
    Write-Host ""

    # Extract data.zip
    if (Test-Path $dataZipPath) {
        Write-Host "Extracting data.zip to destination folder..."
        Expand-Archive -Path $dataZipPath -DestinationPath $DestPath -Force
    } else {
        Write-Host "Warning: data.zip not found in the archive."
    }
    
    # Copy other files (dataset.csv, exceptions.txt, etc.) to destination
    Write-Host "Copying metadata files to destination folder..."
    $filesToCopy = @("README.txt", "dataset.csv", "SHA256SUMS.txt")
    
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
