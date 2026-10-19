#!/bin/bash
#SBATCH --job-name=unzip_wbd
#SBATCH --output=data/slurm/logs/unzip_wbd_%j.out
#SBATCH --error=data/slurm/logs/unzip_wbd_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -e

PROJECT_DIR="/courses/DS5500.202710/teams/team4"
WBD_DIR="$PROJECT_DIR/data/raw/wbd"

cd "$WBD_DIR"

echo "Starting WBD extraction..."
echo "Host: $(hostname)"
echo "Date: $(date)"

unzip -o national_wbd.zip

echo "Extraction complete."

echo "Output file:"
ls -lh WBD_National_GPKG.gpkg

echo "Output file size:"
du -h WBD_National_GPKG.gpkg

# Keep only the HUC8 layer (the only one the pipeline reads) and delete
# the zip, to save space.
cd "$PROJECT_DIR"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal
python -u etl/slim_wbd.py

echo "Finished."
echo "Date: $(date)"