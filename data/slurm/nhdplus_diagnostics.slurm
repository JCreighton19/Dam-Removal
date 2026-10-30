#!/bin/bash
#SBATCH --job-name=nhdplus_diagnostics
#SBATCH --output=data/slurm/logs/nhdplus_diagnostics_%j.out
#SBATCH --error=data/slurm/logs/nhdplus_diagnostics_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Stage 1 of the reach-matching rework: check the stored NHDPlus HR
# relationships (artificial path -> water polygon, catchment, VAA, flow)
# package by package. Attribute tables only; existing outputs untouched.
# Writes data/processed/nhdplus_diagnostics/<case_study>/.
#
#     sbatch data/slurm/nhdplus_diagnostics.slurm                 # every case-study HUC8
#     HU4="0104 0205" sbatch data/slurm/nhdplus_diagnostics.slurm # a subset
#
# Packages slimmed before the water layers were kept are reported as
# checked_without_water_layers; add the layers with
#     HU4="..." sbatch data/slurm/download_streams.slurm

set -euo pipefail

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

python -u etl/nhdplus_diagnostics.py ${HU4:+--hu4 $HU4}

echo "Job completed: $(date)"
