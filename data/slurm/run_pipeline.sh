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
# The case study comes from case_study.env. If its config doesn't exist yet
# and it's a random sample (PILOT_N_DAMS / PILOT_SEED), sample_random_pilot
# is submitted first to generate it.
#
# Not included (run separately):
#   create_env, download_huc8, unzip_huc8   one-time setup
#   report_pilot_eligibility                replaced by build_views; kept until
#                                           the views are checked against it

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
    load_core_tables
    clean_wq_results
    build_views
    build_station_watershed
    download_nlcd
    load_land_cover
    download_climate
    load_climate
    build_analysis_dataset
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
to_submit=("${STAGES[@]:$start}")

if [[ ! -f "config/${DAM_REMOVAL_CASE_STUDY}.yaml" ]]; then
    if [[ -n "${PILOT_N_DAMS:-}" && -n "${PILOT_SEED:-}" \
          && "$DAM_REMOVAL_CASE_STUDY" == "pilot_random${PILOT_N_DAMS}_seed${PILOT_SEED}" ]]; then
        echo "config/${DAM_REMOVAL_CASE_STUDY}.yaml not found; it will be generated first."
        to_submit=(sample_random_pilot "${to_submit[@]}")
    else
        echo "config/${DAM_REMOVAL_CASE_STUDY}.yaml not found (set in data/slurm/case_study.env)."
        exit 1
    fi
fi

mkdir -p data/slurm/logs
echo "Started:    $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"
echo "Submitting from: $from_stage"

previous=""
for stage in "${to_submit[@]}"; do
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
