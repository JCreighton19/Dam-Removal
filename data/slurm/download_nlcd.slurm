#!/bin/bash
#SBATCH --job-name=download_nlcd
#SBATCH --output=data/slurm/logs/download_nlcd_%j.out
#SBATCH --error=data/slurm/logs/download_nlcd_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Downloading Annual NLCD years for eligible stations (run after build_station_watershed)"

# Each year is a ~1.35 GB download, clipped per NHDPlus package and then
# deleted. A re-run skips finished years.
python -u etl/download_land_cover.py

echo "Job completed: $(date)"
