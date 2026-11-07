#!/bin/bash
#SBATCH --job-name=download_climate
#SBATCH --output=data/slurm/logs/download_climate_%j.out
#SBATCH --error=data/slurm/logs/download_climate_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Downloading monthly PRISM climate for eligible stations (run after build_station_watershed)"

# Each month is a ~40 MB download per variable, clipped per NHDPlus package
# and then deleted. A re-run skips finished months. PRISM allows each file
# to be downloaded at most twice per 24 hours.
# ELIGIBILITY_VARIANT=min1yr (or min1yr_2mo): the eligibility sensitivity variant
# (docs/eligibility_sensitivity_prespec.md); unset = the primary rule.
# With it, covers the variant's eligible stations (a superset of the primary's).
python -u etl/download_climate.py ${ELIGIBILITY_VARIANT:+--eligibility-variant "$ELIGIBILITY_VARIANT"}

echo "Job completed: $(date)"
