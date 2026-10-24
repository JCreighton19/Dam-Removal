#!/bin/bash
#SBATCH --job-name=snap_recheck
#SBATCH --output=data/slurm/logs/snap_recheck_%j.out
#SBATCH --error=data/slurm/logs/snap_recheck_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Re-snap a few HUC8s into data/processed/snap_validation/ after a rule
# change, then re-run the comparison over every HUC8 there. By default, the
# N HUC8s with the most stations moved > 300 m by drainage area in the last
# comparison (the 2026-10-24 drainage_area_far fix). Current outputs are
# untouched.
#
#     sbatch data/slurm/snap_recheck.slurm                       # top 5
#     N=10 sbatch data/slurm/snap_recheck.slurm
#     HUC8S="01090002 01100005" sbatch data/slurm/snap_recheck.slurm

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

if [[ -z "${HUC8S:-}" ]]; then
    HUC8S=$(python - "$OUT_DIR/snap_comparison.csv" "${N:-5}" <<'PY'
import sys
import pandas as pd

path, n = sys.argv[1], int(sys.argv[2])
c = pd.read_csv(path, dtype={"feature_id": str, "huc8": str}, low_memory=False)
moved = c[(c["change"] == "moved_reach") & (c["snap_reason"] == "drainage_area_match")
          & (c["snap_distance_m"] - c["old_distance_m"] > 300)]
print(" ".join(moved["huc8"].value_counts().head(n).index))
PY
)
fi
if [[ -z "$HUC8S" ]]; then
    echo "No HUC8s to recheck." >&2
    exit 1
fi
echo "Rechecking HUC8s: $HUC8S"

python -u etl/snap_features.py --output-dir "$OUT_DIR" --huc8 $HUC8S
python -u etl/compare_snapping.py --new-dir "$OUT_DIR" --audit-dir "$AUDIT_DIR"

echo "Job completed: $(date)"
