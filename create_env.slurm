#!/bin/bash
#SBATCH --job-name=create_env
#SBATCH --output=data/slurm/logs/env_%j.out
#SBATCH --error=data/slurm/logs/env_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -e

cd /courses/DS5500.202710/teams/team4

module load anaconda3/2024.06

echo "Starting environment creation: $(date)"
conda env create -f environment.yml
echo "Finished environment creation: $(date)"