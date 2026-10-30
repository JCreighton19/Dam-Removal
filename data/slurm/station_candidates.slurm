#!/bin/bash
#SBATCH --job-name=station_candidates
#SBATCH --output=data/slurm/logs/station_candidates_%j.out
#SBATCH --error=data/slurm/logs/station_candidates_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

# Stage 2 of the reach-matching rework: candidate reaches and their
# evidence for every candidate station (no reach is chosen; current
# snapping outputs untouched). Writes
# data/processed/station_candidates/<case_study>/.
#
#     sbatch data/slurm/station_candidates.slurm                  # every case-study HUC8
#     HU4="0104 0205" sbatch data/slurm/station_candidates.slurm  # a subset
#
# A subset also gets the HUC8s holding the known wrong-reach stations
# (etl/compare_snapping.py KNOWN_WRONG_STATIONS and the audit flags).

set -euo pipefail

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

# PROJ fetches datum grids (e.g. NADCON for NAD27) it needs; see snap_features.slurm.
export PROJ_NETWORK=ON

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

KNOWN=""
if [[ -n "${HU4:-}" ]]; then
    AUDIT_DIR="data/processed/preprocessing_audit/${DAM_REMOVAL_CASE_STUDY}"
    KNOWN=$(python etl/compare_snapping.py --print-known-huc8s --audit-dir "$AUDIT_DIR" || true)
    echo "HUC8s with known wrong-reach stations: ${KNOWN:-none}"
fi

python -u etl/station_candidates.py ${HU4:+--hu4 $HU4} ${KNOWN:+--huc8 $KNOWN}

echo "Job completed: $(date)"
