#!/bin/bash
#SBATCH --job-name=build_analysis_dataset
#SBATCH --output=data/slurm/logs/build_analysis_dataset_%j.out
#SBATCH --error=data/slurm/logs/build_analysis_dataset_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Building the analysis dataset (run after load_climate)"

python -u etl/build_analysis_dataset.py

echo "Job completed: $(date)"
