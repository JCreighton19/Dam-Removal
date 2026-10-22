#!/bin/bash
# Submit the whole pipeline for the case study in case_study.env as a chain
# of Slurm jobs. Each job waits for the previous one to succeed
# (--dependency=afterok); if one fails, every later job is cancelled
# (--kill-on-invalid-dep=yes) instead of waiting in the queue forever.
#
# Run on the login node, from anywhere:
#     bash data/slurm/run_pipeline.sh                    # every stage
#     bash data/slurm/run_pipeline.sh --from snap_features
#     bash data/slurm/run_pipeline.sh --to select_candidate_stations
#     bash data/slurm/run_pipeline.sh --from snap_features --to link_dams
#     bash data/slurm/run_pipeline.sh --list             # show stages
#
# The case study comes from case_study.env. If its config doesn't exist yet
# and it's a sample (PILOT_N_DAMS / PILOT_SEED, or PILOT_N_DAMS=all),
# sample_random_pilot is submitted first to generate it.
#
# 8-hour job limit: stages that resume where they stopped (skipping
# finished files or HUC8s) are submitted as several attempts (ATTEMPTS
# below). Each extra attempt starts when the previous one ends, even if it
# hit the time limit, and finishes in minutes once everything is done. The
# next stage waits for the last attempt to succeed. A stage's HUC8 progress
# markers (data/processed/_progress/<case_study>/<stage>) are cleared when
# the stage is submitted, so a new submission recomputes it.
#
# Not included (run separately):
#   create_env, download_huc8, unzip_huc8   one-time setup
#   build_stream_network                    diagnostics only; nothing reads
#                                           its output
#   report_pilot_eligibility                replaced by build_views; kept until
#                                           the views are checked against it

set -euo pipefail

STAGES=(
    download_wqp_stations
    select_candidate_stations
    download_streams
    unzip_nhdplus
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

# Attempts per stage (1 if not listed). Only for stages that resume.
attempts_for() {
    case "$1" in
        download_wqp_stations) echo 2 ;;
        download_streams)      echo 3 ;;
        unzip_nhdplus)         echo 3 ;;
        snap_features)         echo 3 ;;
        link_dams)             echo 3 ;;
        download_wqp_results)  echo 8 ;;
        download_nlcd)         echo 4 ;;
        load_land_cover)       echo 3 ;;
        download_climate)      echo 5 ;;
        *)                     echo 1 ;;
    esac
}

# sbatch sometimes fails with "Unexpected message received" even though the
# job WAS queued. Each submission carries a unique --comment; on an error,
# the queue is checked for it before retrying, so nothing is submitted twice.
submit() {
    local tag="run_pipeline_$$_$1_$2" attempt id
    shift 2
    for attempt in 1 2 3 4 5; do
        if id=$(sbatch --parsable --comment="$tag" "$@" 2>/dev/null); then
            echo "${id%%;*}"; return 0
        fi
        sleep 10
        id=$(squeue -u "$USER" -h -o "%i %k" | awk -v t="$tag" '$2 == t {print $1; exit}')
        if [[ -n "$id" ]]; then
            echo "$id"; return 0
        fi
        echo "  (sbatch failed for $tag; retrying)" >&2
    done
    echo "sbatch kept failing for $tag; stopping. Jobs above are queued." >&2
    return 1
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

from_stage="${STAGES[0]}"
to_stage="${STAGES[${#STAGES[@]}-1]}"
while (( $# )); do
    case "$1" in
        --list)
            for stage in "${STAGES[@]}"; do
                printf '%-28s attempts: %s\n' "$stage" "$(attempts_for "$stage")"
            done
            exit 0 ;;
        --from) from_stage="${2:?--from needs a stage name (see --list)}"; shift 2 ;;
        --to)   to_stage="${2:?--to needs a stage name (see --list)}"; shift 2 ;;
        *) echo "Unknown argument: $1 (use --from <stage>, --to <stage> or --list)"; exit 1 ;;
    esac
done

stage_index() {
    local i
    for i in "${!STAGES[@]}"; do
        [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return; }
    done
    echo "Unknown stage '$1'. Stages:" >&2; printf '  %s\n' "${STAGES[@]}" >&2; exit 1
}
start=$(stage_index "$from_stage")
end=$(stage_index "$to_stage")
if (( end < start )); then
    echo "--to $to_stage comes before --from $from_stage"; exit 1
fi

source data/slurm/case_study.env
to_submit=("${STAGES[@]:$start:$((end - start + 1))}")

if [[ ! -f "config/${DAM_REMOVAL_CASE_STUDY}.yaml" ]]; then
    if [[ "$PILOT_N_DAMS" == "all" && "$DAM_REMOVAL_CASE_STUDY" == "all_removals" ]] \
       || [[ "$DAM_REMOVAL_CASE_STUDY" == "pilot_random${PILOT_N_DAMS}_seed${PILOT_SEED}" ]]; then
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
echo "Submitting: $from_stage -> $to_stage"

previous=""   # last job of the previous stage
for stage in "${to_submit[@]}"; do
    rm -rf "data/processed/_progress/${DAM_REMOVAL_CASE_STUDY}/${stage}"
    n=$(attempts_for "$stage")
    jobs=()
    attempt_job=""
    for (( attempt = 1; attempt <= n; attempt++ )); do
        # Attempt 1 waits for the previous stage to succeed; later attempts
        # also wait for the previous attempt to end, however it ended.
        dependency=()
        deps=()
        [[ -n "$previous" ]] && deps+=("afterok:$previous")
        [[ -n "$attempt_job" ]] && deps+=("afterany:$attempt_job")
        if (( ${#deps[@]} )); then
            dependency=(--dependency="$(IFS=,; echo "${deps[*]}")" --kill-on-invalid-dep=yes)
        fi
        # ${arr[@]+...} keeps an empty array safe under `set -u` on older bash.
        attempt_job=$(submit "$stage" "$attempt" ${dependency[@]+"${dependency[@]}"} "data/slurm/${stage}.slurm")
        jobs+=("$attempt_job")
    done
    previous="$attempt_job"
    printf '  %-28s job(s) %s\n' "$stage" "${jobs[*]}"
done

echo
echo "Watch:   squeue -u \$USER"
echo "Cancel:  scancel <job ids above>"
echo "Logs:    data/slurm/logs/"
