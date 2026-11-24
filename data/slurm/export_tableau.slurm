#!/bin/bash
#SBATCH --job-name=export_tableau
#SBATCH --output=data/slurm/logs/export_tableau_%j.out
#SBATCH --error=data/slurm/logs/export_tableau_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1

# Small CSV extracts for the Tableau dashboard (analysis/export_tableau.py).
# Read-only: reads the warehouse, the frozen model outputs and the predictive
# outputs; writes only data/processed/tableau/<case_study>/ (a few MB).
#
#   sbatch data/slurm/export_tableau.slurm
# Models from a specific folder (default: the 2026-11-18 frozen snapshot copy
# if it has one, else the live data/processed/mixed_model/<case_study>/):
#   MODEL_DIR=data/processed/frozen/models-frozen-2026-11-18/all_removals sbatch data/slurm/export_tableau.slurm
#
# Then copy to the laptop (run there):
#   rsync -avz -e "ssh -i ~/Northeastern/ssh/<key>" \
#     <user>@<explorer-login>:/courses/DS5500.202710/teams/team4/data/processed/tableau/all_removals/ \
#     ~/DS5500/Dam-Removal/tableau_data/

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

ARGS=()
if [[ -n "${MODEL_DIR:-}" ]]; then
    ARGS+=(--model-dir "$MODEL_DIR")
fi

python -u analysis/export_tableau.py "${ARGS[@]}"

echo "Job completed: $(date)"
