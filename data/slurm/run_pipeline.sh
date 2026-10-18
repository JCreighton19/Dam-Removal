#!/bin/bash
# Submit the whole pipeline for the case study in case_study.env as a chain
# of Slurm jobs. Each job waits for the previous one to succeed
# (--dependency=afterok); if one fails, every later job is cancelled
# (--kill-on-invalid-dep=yes) instead of waiting in the queue forever.
#
# Run on the login node, from anywhere:
#     bash data/slurm/run_pipeline.sh                    # every stage
#     bash data/slurm/run_pipeline.sh --from snap_features
#     bash data/slurm/run_pipeline.sh --list             # show stages
#
# Not included (run separately):
#   create_env, download_huc8, unzip_huc8   one-time setup
#   sample_random_pilot                     creates the config this reads
#   download_nlcd                           context data, not yet wired in

set -euo pipefail

STAGES=(
    download_wqp_stations
    select_candidate_stations
    download_streams
    unzip_nhdplus
    build_stream_network
    snap_features
    link_dams
    download_wqp_results
    report_pilot_eligibility
    load_core_tables
    clean_wq_results
)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

from_stage="${STAGES[0]}"
case "${1:-}" in
    --list) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
    --from) from_stage="${2:?--from needs a stage name (see --list)}" ;;
    "") ;;
    *) echo "Unknown argument: $1 (use --from <stage> or --list)"; exit 1 ;;
esac

start=-1
for i in "${!STAGES[@]}"; do
    [[ "${STAGES[$i]}" == "$from_stage" ]] && start=$i
done
if (( start < 0 )); then
    echo "Unknown stage '$from_stage'. Stages:"; printf '  %s\n' "${STAGES[@]}"; exit 1
fi

source data/slurm/case_study.env
if [[ ! -f "config/${DAM_REMOVAL_CASE_STUDY}.yaml" ]]; then
    echo "config/${DAM_REMOVAL_CASE_STUDY}.yaml not found (set in data/slurm/case_study.env)."
    exit 1
fi

mkdir -p data/slurm/logs
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Submitting from: $from_stage"

previous=""
for stage in "${STAGES[@]:$start}"; do
    dependency=()
    [[ -n "$previous" ]] && dependency=(--dependency="afterok:$previous" --kill-on-invalid-dep=yes)
    # ${arr[@]+...} keeps an empty array safe under `set -u` on older bash.
    previous=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} "data/slurm/${stage}.slurm")
    printf '  %-28s job %s\n' "$stage" "$previous"
done

echo
echo "Watch:   squeue -u \$USER"
echo "Cancel:  scancel <job ids above>"
echo "Logs:    data/slurm/logs/"
