#!/bin/bash
#SBATCH --job-name=sample_random_pilot
#SBATCH --output=data/slurm/logs/sample_random_pilot_%j.out
#SBATCH --error=data/slurm/logs/sample_random_pilot_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Starting random pilot sampling"
echo "Start time: $(date)"

# Sample size and seed come from case_study.env (PILOT_N_DAMS, PILOT_SEED).
# run_pipeline.sh submits this job automatically when the config is missing.
python -u etl/sample_random_pilot.py --n-dams "$PILOT_N_DAMS" --seed "$PILOT_SEED"

echo "Finished: $(date)"
