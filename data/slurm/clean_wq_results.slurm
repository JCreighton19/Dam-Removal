#!/bin/bash
#SBATCH --job-name=clean_wq_results
#SBATCH --output=data/slurm/logs/clean_wq_results_%j.out
#SBATCH --error=data/slurm/logs/clean_wq_results_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Cleaning WQP results into wq_measurement (run after load_core_tables)"

python -u etl/clean_wq_results.py

echo "Job completed: $(date)"
