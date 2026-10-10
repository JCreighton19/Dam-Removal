#!/bin/bash
#SBATCH --job-name=wqp-stations
#SBATCH --output=logs/wqp_stations_%j.out
#SBATCH --error=logs/wqp_stations_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

source ~/.bashrc
conda activate dam-removal

echo "Starting WQP station pilot"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

python -u etl/download_wqp_stations.py \
    --n-hucs 10

echo "End time: $(date)"