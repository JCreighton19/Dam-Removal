#!/bin/bash
#SBATCH --job-name=build_network
#SBATCH --output=data/slurm/logs/build_network_%j.out
#SBATCH --error=data/slurm/logs/build_network_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -e

cd /courses/DS5500.202710/teams/team4

module load anaconda3/2024.06

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "========================================"
echo "Build Network job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "========================================"

python -u etl/build_stream_network.py

echo "Build Network job finished: $(date)"