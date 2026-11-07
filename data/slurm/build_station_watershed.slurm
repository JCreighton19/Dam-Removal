#!/bin/bash
#SBATCH --job-name=build_station_watershed
#SBATCH --output=data/slurm/logs/build_station_watershed_%j.out
#SBATCH --error=data/slurm/logs/build_station_watershed_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Building upstream areas for eligible stations (run after build_views)"

# Add --with-outlines to also build map outlines for QA (slow for large
# rivers; later stages don't use them).
# ELIGIBILITY_VARIANT=min1yr (or min1yr_2mo): the eligibility sensitivity variant
# (docs/eligibility_sensitivity_prespec.md); unset = the primary rule.
# With it, covers the variant's eligible stations (a superset of the primary's).
python -u etl/build_station_watershed.py ${ELIGIBILITY_VARIANT:+--eligibility-variant "$ELIGIBILITY_VARIANT"}

echo "Job completed: $(date)"
