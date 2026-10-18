#!/bin/bash
#SBATCH --job-name=build_station_watershed
#SBATCH --output=data/slurm/logs/build_station_watershed_%j.out
#SBATCH --error=data/slurm/logs/build_station_watershed_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Building upstream areas for eligible stations (run after build_views)"

# --with-outlines builds map outlines for QA; drop it at national scale
# (slow for large rivers, and later stages don't use them).
python -u etl/build_station_watershed.py --with-outlines

echo "Job completed: $(date)"
