#!/bin/bash
#SBATCH --job-name=load_climate
#SBATCH --output=data/slurm/logs/load_climate_%j.out
#SBATCH --error=data/slurm/logs/load_climate_%j.err
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
echo "Summarizing monthly climate per station (run after download_climate)"

python -u etl/load_climate.py

echo "Job completed: $(date)"
