#!/bin/bash
#SBATCH --job-name=build_daily_measurements
#SBATCH --output=data/slurm/logs/build_daily_measurements_%j.out
#SBATCH --error=data/slurm/logs/build_daily_measurements_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Daily aggregation of wq_measurement (run after clean_wq_results)"

python -u etl/build_daily_measurements.py

echo "Job completed: $(date)"
