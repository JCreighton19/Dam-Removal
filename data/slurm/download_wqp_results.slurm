#!/bin/bash
#SBATCH --job-name=wqp_results
#SBATCH --output=data/slurm/logs/wqp_results_%j.out
#SBATCH --error=data/slurm/logs/wqp_results_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

# Activate your existing environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Starting WQP result download"
date

python -u etl/download_wqp_results.py \
    --max-stations 50 \
    --batch-size 5

echo "Finished WQP result download"
date