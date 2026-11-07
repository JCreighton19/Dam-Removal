#!/bin/bash
#SBATCH --job-name=fit_mixed_model
#SBATCH --output=data/slurm/logs/mm_%j.out
#SBATCH --error=data/slurm/logs/mm_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Fits the mixed-effects removal models (analysis/mixed_model.py) on
# analysis_measurement and writes data/processed/mixed_model/<case_study>/
# (read-only on the warehouse; run after build_analysis_dataset).
# Results: notebooks/analysis/mixed_model_results.ipynb.

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

# INPUT_VARIANT=min1yr (or min1yr_2mo): fit the same models on
# analysis_measurement_<variant>.parquet (eligibility sensitivity analysis,
# docs/eligibility_sensitivity_prespec.md) into
# data/processed/mixed_model/<case_study>/sensitivity_<variant>/mixed_model/; the
# primary outputs are not touched.
if [[ -n "${INPUT_VARIANT:-}" ]]; then
    input="data/warehouse/$DAM_REMOVAL_CASE_STUDY/analysis_measurement_${INPUT_VARIANT}.parquet"
    [[ -f "$input" ]] || { echo "No $input (run build_analysis_dataset with ELIGIBILITY_VARIANT=$INPUT_VARIANT)"; exit 1; }
    echo "Sensitivity variant $INPUT_VARIANT: $input"
    python -u analysis/mixed_model.py --input "$input" \
        --output-dir "data/processed/mixed_model/$DAM_REMOVAL_CASE_STUDY/sensitivity_${INPUT_VARIANT}/mixed_model"
else
    python -u analysis/mixed_model.py
fi

echo "Job completed: $(date)"
