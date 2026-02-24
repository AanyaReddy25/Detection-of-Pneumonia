# ============================================================
# Download Chest X-Ray Pneumonia Dataset from Kaggle (Windows)
# ============================================================
# Prerequisites:
#   1. Install Kaggle CLI:  pip install kaggle
#   2. Place your kaggle.json API key at %USERPROFILE%\.kaggle\kaggle.json
#      (Download from https://www.kaggle.com/settings -> API -> Create New Token)
#
# Usage:
#   .\download_dataset.ps1
# ============================================================

$Dataset = "paultimothymooney/chest-xray-pneumonia"
$TargetDir = "code\data\input"

Write-Host "============================================"
Write-Host " Downloading Chest X-Ray Pneumonia Dataset"
Write-Host "============================================"

# Check if kaggle CLI is available
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'kaggle' CLI not found." -ForegroundColor Red
    Write-Host "Install it with: pip install kaggle"
    Write-Host "Then place your API key at $env:USERPROFILE\.kaggle\kaggle.json"
    exit 1
}

# Create target directory
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

# Download dataset
Write-Host "Downloading dataset from Kaggle..."
kaggle datasets download -d $Dataset -p $TargetDir --unzip

Write-Host ""
Write-Host "============================================"
Write-Host " Download complete!"
Write-Host " Dataset extracted to: $TargetDir\"
Write-Host "============================================"
Write-Host ""
Write-Host "If the data is nested under chest_xray\, move it up:"
Write-Host "  Move-Item $TargetDir\chest_xray\* $TargetDir\"
Write-Host "  Remove-Item $TargetDir\chest_xray"
