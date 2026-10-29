#!/bin/bash
#SBATCH --job-name=fit_mixed_model
#SBATCH --output=data/slurm/logs/fit_mixed_model_%j.out
#SBATCH --error=data/slurm/logs/fit_mixed_model_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Fits the mixed-effects removal models (analysis/mixed_model.py) on
# analysis_measurement and writes data/processed/mixed_model/<case_study>/
# (read-only on the warehouse; run after build_analysis_dataset).
# Results: notebooks/mixed_model_results.ipynb.

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

python -u analysis/mixed_model.py

echo "Job completed: $(date)"
