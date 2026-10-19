#!/bin/bash
#SBATCH --job-name=load_land_cover
#SBATCH --output=data/slurm/logs/load_land_cover_%j.out
#SBATCH --error=data/slurm/logs/load_land_cover_%j.err
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
echo "Summarizing land cover per station and year (run after download_nlcd)"

python -u etl/load_land_cover.py

echo "Job completed: $(date)"
