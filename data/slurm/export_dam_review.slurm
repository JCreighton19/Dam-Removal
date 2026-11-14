#!/bin/bash
#SBATCH --job-name=export_dam_review
#SBATCH --output=data/slurm/logs/export_dam_review_%j.out
#SBATCH --error=data/slurm/logs/export_dam_review_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2

# Export the dams to review by hand, with the flowlines around them, as one
# GeoPackage for QGIS (analysis/audits/export_dam_review.py; read-only;
# writes data/processed/dam_review/<case_study>/). Workflow: docs/dam_review.md.
#
#   sbatch data/slurm/export_dam_review.slurm
#   RADIUS_KM=4 sbatch data/slurm/export_dam_review.slurm          # wider circles
#   DAMS="CO-022 WI-043" RADIUS_KM=6 sbatch data/slurm/export_dam_review.slurm
#        (those dams only -> dam_review_extra.gpkg; the main export is kept)

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; radius: ${RADIUS_KM:-3} km; dams: ${DAMS:-review set}"

# DAMS is split into words on purpose.
# shellcheck disable=SC2086
python -u analysis/audits/export_dam_review.py --radius-km "${RADIUS_KM:-3}" \
    ${DAMS:+--dams $DAMS}

echo "Job completed: $(date)"
