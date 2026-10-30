#!/bin/bash
#SBATCH --job-name=download_nhdplus
#SBATCH --output=data/slurm/logs/nhdplus_%j.out
#SBATCH --error=data/slurm/logs/nhdplus_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -e

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "========================================"
echo "NHDPlus download job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "========================================"

# HU4="0104 0205" sbatch data/slurm/download_streams.slurm limits the run
# to the case study's HUC8s in those HUC4s. Packages slimmed before the
# water layers (NHDArea, NHDWaterbody, NHDPlusBurnWaterbody) were kept are
# downloaded again and only those layers are added.
python -u etl/download_nhdplus.py ${HU4:+--hu4 $HU4}

echo "NHDPlus download job finished: $(date)"