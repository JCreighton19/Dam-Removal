#!/bin/bash
#SBATCH --job-name=pilot_eligibility
#SBATCH --output=data/slurm/logs/pilot_eligibility_%j.out
#SBATCH --error=data/slurm/logs/pilot_eligibility_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "========================================"
echo "Pilot eligibility report job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "========================================"

python -u etl/report_pilot_eligibility.py

echo "Pilot eligibility report job finished: $(date)"
