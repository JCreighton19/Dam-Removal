#!/bin/bash
#SBATCH --job-name=load_core_tables
#SBATCH --output=data/slurm/logs/load_core_tables_%j.out
#SBATCH --error=data/slurm/logs/load_core_tables_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Loading core warehouse tables"

python -u etl/load_core_tables.py

echo "Job completed: $(date)"
