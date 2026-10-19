#!/bin/bash
#SBATCH --job-name=unzip_nhdplus
#SBATCH --output=data/slurm/logs/unzip_nhdplus_%j.out
#SBATCH --error=data/slurm/logs/unzip_nhdplus_%j.err
#SBATCH --time=08:00:00
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
echo "Extracting NHDPlus packages to slim GeoPackages (run after download_streams)"

# Keeps only the 4 layers the pipeline reads, then deletes the extracted
# geodatabase and the zip. A re-run skips finished packages.
python -u etl/unzip_nhdplus.py

echo "Job completed: $(date)"
