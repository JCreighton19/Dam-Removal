#!/bin/bash
#SBATCH --job-name=select_station_reaches
#SBATCH --output=data/slurm/logs/select_station_reaches_%j.out
#SBATCH --error=data/slurm/logs/select_station_reaches_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Stage 3 of the reach-matching rework: choose each station's reach from
# its stage-2 candidates (run station_candidates.slurm first), apply
# config/snap_overrides.csv, place accepted stations on their reach, then
# check the known wrong-reach stations (config/known_station_reaches.csv).
# Writes data/processed/station_reaches/<case_study>/; current snapping
# outputs untouched.
#
#     sbatch data/slurm/select_station_reaches.slurm
#     HU4="0104 0205" sbatch data/slurm/select_station_reaches.slurm
#
# First, once: record the known wrong reaches from the old snapping
# outputs into config/known_station_reaches.csv (review, then commit it):
#     FILL_WRONG_REACHES=1 sbatch data/slurm/select_station_reaches.slurm
#
# The known-station check fails the job if a station is accepted on its
# known wrong reach. Passing it does not show the stations are matched
# correctly; see etl/check_known_stations.py.

set -euo pipefail

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

if [[ -n "${FILL_WRONG_REACHES:-}" ]]; then
    python -u etl/check_known_stations.py --fill-wrong-reaches
    echo "Job completed: $(date)"
    exit 0
fi

python -u etl/select_station_reaches.py ${HU4:+--hu4 $HU4}

echo
echo "=== Known wrong-reach stations ==="
python -u etl/check_known_stations.py

echo "Job completed: $(date)"
