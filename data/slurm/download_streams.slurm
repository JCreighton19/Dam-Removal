#!/bin/bash
#SBATCH --job-name=download_nhdplus
#SBATCH --output=data/slurm/logs/nhdplus_%j.out
#SBATCH --error=data/slurm/logs/nhdplus_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

echo "========================================"
echo "NHDPlus download job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module load python

cd /courses/DS5500.202710/teams/team4

echo "Working directory: $(pwd)"
echo "Running Python downloader..."

python etl/download_nhdplus.py

echo "========================================"
echo "NHDPlus download job finished"
echo "End time: $(date)"
echo "Exit code: $?"
echo "========================================"