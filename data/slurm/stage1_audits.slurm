#!/bin/bash
#SBATCH --job-name=stage1_audits
#SBATCH --output=data/slurm/logs/stage1_audits_%j.out
#SBATCH --error=data/slurm/logs/stage1_audits_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Stage 1 read-only audits (reservoir size; downstream coverage by distance
# band). They read the existing warehouse and processed outputs and write
# only to data/processed/stage1_audit/<case_study>/; no ETL stage is rerun.
#
#   AUDIT=both (default) | reservoir | coverage
#   STORAGE_FILE=<path>   optional NID / NABD export for the reservoir audit
#   STORAGE_ARGS="..."    extra flags, e.g. "--id-column NIDID --storage-column NORM_STOR"
#
#   sbatch data/slurm/stage1_audits.slurm
#   AUDIT=reservoir STORAGE_FILE=data/raw/nid/nation.csv sbatch data/slurm/stage1_audits.slurm

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

AUDIT="${AUDIT:-both}"
echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; audit: $AUDIT"

if [[ "$AUDIT" == "both" || "$AUDIT" == "coverage" ]]; then
    echo "=== Downstream coverage audit ==="
    python -u analysis/audits/downstream_coverage_audit.py --threads 4 --memory-limit 18GB
fi

if [[ "$AUDIT" == "both" || "$AUDIT" == "reservoir" ]]; then
    echo "=== Reservoir-size audit ==="
    # STORAGE_ARGS is split into words on purpose.
    # shellcheck disable=SC2086
    python -u analysis/audits/reservoir_size_audit.py \
        ${STORAGE_FILE:+--storage-file "$STORAGE_FILE"} ${STORAGE_ARGS:-}
fi

echo "Job completed: $(date)"
