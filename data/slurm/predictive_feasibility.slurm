#!/bin/bash
#SBATCH --job-name=predictive_feasibility
#SBATCH --output=data/slurm/logs/pf_%j.out
#SBATCH --error=data/slurm/logs/pf_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Predictive-feasibility analysis (analysis/predictive_feasibility.py;
# docs/predictive_feasibility_prespec.md). Post-hoc and read-only with respect
# to the frozen models (docs/model_freeze.md): reads analysis_measurement, the
# warehouse dam table and data/processed/mixed_model/<case>/ (results.csv,
# dam_effects.csv), writes only data/processed/predictive/<case_study>/.
# Results: notebooks/analysis/predictive_feasibility.ipynb.
#
# STAGE=audit (default): Stage 1 data audit, no targets or models.
#   sbatch data/slurm/predictive_feasibility.slurm

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

STAGE="${STAGE:-audit}"
echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY  Stage: $STAGE"

python -u analysis/predictive_feasibility.py "$STAGE"

echo "Job completed: $(date)"
