#!/bin/bash
#SBATCH --job-name=fit_stage4
#SBATCH --output=data/slurm/logs/stage4_%j.out
#SBATCH --error=data/slurm/logs/stage4_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Stage 4: influence of individual dams on the primary BACI distance models
# (leave-one-dam-out, dam bootstrap) and the pH upstream / downstream
# components (analysis/influence.py), then the consolidated tables and
# figures (analysis/stage4_report.py). Read-only on the warehouse; writes
# data/processed/mixed_model/<case_study>/stage4/{influence,report}/.
# Run after fit_distance_models.slurm (RE=auto, plus RE=intercepts and
# RE=slopes for the comparisons), e.g.
#
#   a=$(sbatch --parsable data/slurm/fit_distance_models.slurm)
#   b=$(RE=intercepts RUNS="primary size_proxy" sbatch --parsable data/slurm/fit_distance_models.slurm)
#   c=$(RE=slopes RUNS="primary size_proxy" sbatch --parsable data/slurm/fit_distance_models.slurm)
#   sbatch --dependency=afterok:$a:$b:$c data/slurm/fit_stage4.slurm
#
#   BOOT=200     bootstrap draws (0 skips the bootstrap)
#   REPORT_ONLY=1  rebuild the report only (e.g. after changing SESOI)

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; bootstrap draws: ${BOOT:-200}; cpus: ${SLURM_CPUS_PER_TASK:-1}"

if [[ "${REPORT_ONLY:-0}" != "1" ]]; then
    echo "=== Influence (leave-one-dam-out, bootstrap, BACI components) ==="
    python -u analysis/influence.py --re auto --bootstrap "${BOOT:-200}" \
        --workers "${SLURM_CPUS_PER_TASK:-1}" --components-series ph water_temp dissolved_oxygen
fi

echo "=== Report ==="
python -u analysis/stage4_report.py

echo "Job completed: $(date)"
