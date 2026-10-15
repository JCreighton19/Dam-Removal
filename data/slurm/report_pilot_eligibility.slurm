#!/bin/bash
#SBATCH --job-name=pilot_eligibility
#SBATCH --output=data/slurm/logs/pilot_eligibility_%j.out
#SBATCH --error=data/slurm/logs/pilot_eligibility_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "========================================"
echo "Pilot eligibility report job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "========================================"

python -u etl/report_pilot_eligibility.py --case-study pilot_19020401 --huc8 19020401

echo "Pilot eligibility report job finished: $(date)"
