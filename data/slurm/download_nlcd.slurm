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

mkdir -p data/slurm/logs
mkdir -p data/raw/nlcd

echo "Starting NLCD download..."
echo "Project: $PROJECT_DIR"
echo "Output: $PROJECT_DIR/data/raw/nlcd"
echo "Started: $(date)"

# Load Conda and initialize shell integration.
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the existing environment.
conda activate dam-removal

# Verify the required package is available.
python -c "import rasterio; print('Rasterio:', rasterio.__version__)"

# Run the download.
python -u etl/download_land_cover.py

echo "Finished: $(date)"