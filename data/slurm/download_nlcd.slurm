#!/bin/bash
#SBATCH --job-name=nlcd_download
#SBATCH --output=data/slurm/logs/nlcd_%j.out
#SBATCH --error=data/slurm/logs/nlcd_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

PROJECT_DIR="/courses/DS5500.202710/teams/team4"

cd "$PROJECT_DIR"

source data/slurm/case_study.env
mkdir -p data/raw/nlcd

echo "Starting NLCD download..."
echo "Project: $PROJECT_DIR"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Output: $PROJECT_DIR/data/raw/nlcd"
echo "Started: $(date)"

# Load Conda and initialize shell integration.
# Use the verified Conda environment directly.
export CONDA_ENV="/home/creighton.jo/.conda/envs/dam-removal"
export PATH="$CONDA_ENV/bin:$PATH"

# Verify the required package is available.
python -c "import rasterio; print('Rasterio:', rasterio.__version__)"

# Run the download.
python -u etl/download_land_cover.py

echo "Finished: $(date)"