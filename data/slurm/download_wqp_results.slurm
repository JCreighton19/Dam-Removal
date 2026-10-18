#!/bin/bash
#SBATCH --job-name=wqp_results
#SBATCH --output=data/slurm/logs/wqp_results_%j.out
#SBATCH --error=data/slurm/logs/wqp_results_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

# Activate your existing environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Starting WQP result download"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "NOTE: must run after link_dams.slurm -- this now downloads results"
echo "only for stations link_dams_stations.py confirmed network-connected."
date

python -u etl/download_wqp_results.py \
    --batch-size 5 \
    --workers 2

echo "Finished WQP result download"
date
