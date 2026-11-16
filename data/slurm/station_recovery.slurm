#!/bin/bash
#SBATCH --job-name=station_recovery
#SBATCH --output=data/slurm/logs/station_recovery_%j.out
#SBATCH --error=data/slurm/logs/station_recovery_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2

# Expanded station recovery audit (analysis/audits/station_recovery_audit.py):
# how much useful data the unmatched stations could add. Read-only for the
# pipeline; writes data/processed/stage1_audit/<case_study>/station_recovery/
# and (summary stage) data/raw/wqp/summary/<case_study>/station_recovery/.
#
#   STAGE=structure   no network: unmatched population, reasons, regions,
#                     distances, roles, dam states
#   STAGE=summary     WQP summary counts (station x year x characteristic) for
#                     the ambiguous stations linked within the limits; counts
#                     only, no result rows. Re-run to retry failed batches.
#   STAGE=estimate    eligibility and per-parameter gains, calibrated on the
#                     181 stations already downloaded; priority list
#   STAGE=all         structure, summary, estimate
#
#   STAGE=structure sbatch data/slurm/station_recovery.slurm

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

STAGE="${STAGE:-structure}"
echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; stage: $STAGE"

for stage in structure summary estimate; do
    if [[ "$STAGE" == "all" || "$STAGE" == "$stage" ]]; then
        echo "=== $stage ==="
        python -u analysis/audits/station_recovery_audit.py --stage "$stage"
    fi
done

echo "Job completed: $(date)"
