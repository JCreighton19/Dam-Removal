#!/bin/bash
#SBATCH --job-name=match_dams
#SBATCH --output=data/slurm/logs/match_dams_%j.out
#SBATCH --error=data/slurm/logs/match_dams_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

# Stage 4 of the reach-matching rework: match every dam removal to its
# reservoir (if mapped), structure flowline and the flowlines below it;
# apply config/snap_overrides.csv (feature_kind dam). Writes
# data/processed/dam_reaches/<case_study>/; current snapping outputs
# untouched. The log ends with reservoir / outlet counts (how many dams sit
# at a mapped reservoir, outlets per reservoir, diversions).
#
#     sbatch data/slurm/match_dams.slurm                  # every case-study HUC8
#     HU4="0104 0205" sbatch data/slurm/match_dams.slurm  # a subset

set -euo pipefail

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

python -u etl/match_dams.py ${HU4:+--hu4 $HU4}

echo "Job completed: $(date)"
