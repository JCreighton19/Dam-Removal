#!/bin/bash
#SBATCH --job-name=build_analysis_dataset
#SBATCH --output=data/slurm/logs/build_analysis_dataset_%j.out
#SBATCH --error=data/slurm/logs/build_analysis_dataset_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Building the analysis dataset (run after load_climate)"

# ELIGIBILITY_VARIANT=min1yr (or min1yr_2mo): the eligibility sensitivity variant
# (docs/eligibility_sensitivity_prespec.md); unset = the primary rule.
# With it, writes analysis_measurement_<variant>.parquet; the primary table is not touched.
# VERIFY_PRIMARY=1: rebuild the primary table in memory and check it equals the existing
# analysis_measurement.parquet (writes nothing), e.g. after the context stages ran for a variant.
python -u etl/build_analysis_dataset.py ${ELIGIBILITY_VARIANT:+--eligibility-variant "$ELIGIBILITY_VARIANT"} \
    ${VERIFY_PRIMARY:+--verify-primary}

echo "Job completed: $(date)"
