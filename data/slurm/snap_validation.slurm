#!/bin/bash
#SBATCH --job-name=snap_validation
#SBATCH --output=data/slurm/logs/snap_validation_%j.out
#SBATCH --error=data/slurm/logs/snap_validation_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Validate the snapping rules before re-running the pipeline: snap the
# HUC8s holding known / audit-flagged failures plus a random sample into
# data/processed/snap_validation/ (current outputs untouched), then compare
# with the current snapped features. See docs/rerun_after_snapping.md.
#
#     sbatch data/slurm/snap_validation.slurm             # 30 random HUC8s
#     SAMPLE=60 sbatch data/slurm/snap_validation.slurm

set -euo pipefail

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

# PROJ fetches datum grids (e.g. NADCON for NAD27) it needs; see snap_features.slurm.
export PROJ_NETWORK=ON

AUDIT_DIR="data/processed/preprocessing_audit/${DAM_REMOVAL_CASE_STUDY}"
OUT_DIR="data/processed/snap_validation"

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

KNOWN=$(python etl/compare_snapping.py --print-known-huc8s --audit-dir "$AUDIT_DIR")
echo "HUC8s with known failures: ${KNOWN:-none}"

# A failed HUC8 is reported but doesn't stop the comparison of the others.
python -u etl/snap_features.py --output-dir "$OUT_DIR" --sample "${SAMPLE:-30}" \
    ${KNOWN:+--huc8 $KNOWN} || echo "WARNING: some HUC8s failed; see above"
python -u etl/compare_snapping.py --new-dir "$OUT_DIR" --audit-dir "$AUDIT_DIR"

echo "Job completed: $(date)"
