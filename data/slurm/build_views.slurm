#!/bin/bash
#SBATCH --job-name=build_views
#SBATCH --output=data/slurm/logs/build_views_%j.out
#SBATCH --error=data/slurm/logs/build_views_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Building warehouse.duckdb views (run after clean_wq_results)"

python -u etl/build_views.py

echo "Job completed: $(date)"
