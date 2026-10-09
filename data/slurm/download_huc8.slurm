#!/bin/bash
#SBATCH --job-name=wbd_download
#SBATCH --output=logs/wbd_%j.out
#SBATCH --error=logs/wbd_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

PROJECT_DIR="/courses/DS5500.202710/teams/team4"
RAW_DIR="$PROJECT_DIR/data/raw/wbd"

mkdir -p "$RAW_DIR"
mkdir -p "$PROJECT_DIR/logs"

cd "$RAW_DIR"

echo "Starting National WBD download..."
echo "Output: $RAW_DIR"

wget -O national_wbd.zip \
"https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GPKG/WBD_National_GPKG.zip"

echo "Download complete."