#!/bin/bash
#SBATCH --job-name=build_dam_reservoir
#SBATCH --output=data/slurm/logs/build_dam_reservoir_%j.out
#SBATCH --error=data/slurm/logs/build_dam_reservoir_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Mapped reservoir polygon (area, timing) per dam removal, from the Stage 4
# dam_reaches outputs and the extracted NHDPlus HR packages. Writes
# dam_reservoir.parquet to the warehouse (run after build_views, before
# build_analysis_dataset).

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

python -u etl/build_dam_reservoir.py

echo "Job completed: $(date)"
