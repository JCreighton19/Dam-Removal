#!/bin/bash
#SBATCH --job-name=station_payoff
#SBATCH --output=data/slurm/logs/station_payoff_%j.out
#SBATCH --error=data/slurm/logs/station_payoff_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Station payoff, step 2 (analysis/audits/station_payoff_recount.py): list the
# high-value ambiguous stations from the payoff audit (run
# AUDIT=station_payoff with stage1_audits.slurm first), download their WQP
# samples into data/raw/wqp/results/<case_study>/station_payoff/ (not the
# pipeline's pilot_download), and count how many dams they would really add.
# Nothing in the warehouse or the pipeline outputs is changed.
#
#   STAGE=all (default) | list | download | recount
#   GAINS="baci_control first_downstream" (default) which payoff gains to download
#
#   sbatch data/slurm/station_payoff.slurm
#   STAGE=recount sbatch data/slurm/station_payoff.slurm     # after a download re-run
#
# A download stopped by WQP rate limiting exits non-zero; re-run with
# STAGE=download (finished batches are kept), then STAGE=recount.

set -euo pipefail

cd /courses/DS5500.202710/teams/team4

source data/slurm/case_study.env

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

STAGE="${STAGE:-all}"
GAINS="${GAINS:-baci_control first_downstream}"
LIST="data/processed/stage1_audit/$DAM_REMOVAL_CASE_STUDY/station_payoff/download_list.csv"
echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; stage: $STAGE; gains: $GAINS"

if [[ "$STAGE" == "all" || "$STAGE" == "list" ]]; then
    echo "=== List ==="
    # GAINS is split into words on purpose.
    # shellcheck disable=SC2086
    python -u analysis/audits/station_payoff_recount.py --stage list --gains $GAINS
fi

if [[ "$STAGE" == "all" || "$STAGE" == "download" ]]; then
    echo "=== Download ==="
    python -u etl/download_wqp_results.py --stations-file "$LIST" \
        --output-subdir station_payoff --batch-size 5 --workers 2
fi

if [[ "$STAGE" == "all" || "$STAGE" == "recount" ]]; then
    echo "=== Recount ==="
    python -u analysis/audits/station_payoff_recount.py --stage recount
fi

echo "Job completed: $(date)"
