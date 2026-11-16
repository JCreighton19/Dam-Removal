#!/bin/bash
#SBATCH --job-name=export_station_review
#SBATCH --output=data/slurm/logs/export_station_review_%j.out
#SBATCH --error=data/slurm/logs/export_station_review_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2

# Accepted-station accuracy review: every analysis station with its matching
# decision, alternatives, roles, risk tier and the stratified sample, as one
# GeoPackage for QGIS (analysis/audits/export_station_review.py; read-only;
# writes data/processed/station_review/<case_study>/). Workflow:
# docs/station_review.md.
#
#   RADIUS_KM=1 (flowlines around each station), SAMPLE_FRACTION=0.15,
#   MIN_PER_STRATUM=3
#   sbatch data/slurm/export_station_review.slurm

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

python -u analysis/audits/export_station_review.py --radius-km "${RADIUS_KM:-1}" \
    --sample-fraction "${SAMPLE_FRACTION:-0.15}" --min-per-stratum "${MIN_PER_STRATUM:-3}"

echo "Job completed: $(date)"
