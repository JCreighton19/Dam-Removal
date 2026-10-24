#!/bin/bash
#SBATCH --job-name=snap_features
#SBATCH --output=data/slurm/logs/snap_%j.out
#SBATCH --error=data/slurm/logs/snap_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -e

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

# Let PROJ fetch datum grids it needs (e.g. NADCON us_noaa_conus.tif for
# NAD27 stations) from cdn.proj.org; they are cached under ~/.local/share/proj.
# Without them snap_features.py stops with the grid names (it never falls
# back to an approximate shift).
export PROJ_NETWORK=ON

echo "========================================"
echo "Snap Features job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "========================================"

python -u etl/snap_features.py

echo "Snap Features job finished: $(date)"