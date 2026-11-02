#!/bin/bash
#SBATCH --job-name=link_reaches
#SBATCH --output=data/slurm/logs/link_reaches_%j.out
#SBATCH --error=data/slurm/logs/link_reaches_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Stage 5 of the reach-matching rework: link dams (stage 4) to stations
# (stage 3) along the network and compare with the current
# dam_station_links_<huc8>.csv. Writes data/processed/reach_links/<case_study>/;
# current links untouched. The log ends with the old-vs-new comparison:
# pairs kept / gained / lost, what blocks each lost pair (e.g.
# station:ambiguous:confluence_ambiguous), dams with up- and downstream
# stations in the exposure window, and the flowline-direction check.
#
#     sbatch data/slurm/link_reaches.slurm                   # every HUC8
#     HU4="0104 0205" sbatch data/slurm/link_reaches.slurm   # a subset
#     MAX_KM=50 sbatch data/slurm/link_reaches.slurm         # cap each search
#
# RESUME=1 skips HUC8s an earlier attempt finished (run_pipeline.sh sets it
# for the extra attempts after a time-out); without it every chosen HUC8 is
# recomputed.

set -euo pipefail

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

python -u etl/link_reaches.py ${HU4:+--hu4 $HU4} ${MAX_KM:+--max-km $MAX_KM} ${RESUME:+--resume}

echo "Job completed: $(date)"
