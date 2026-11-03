#!/bin/bash
# Speed up a per-HUC8 stage (match_dams or link_reaches) that run_pipeline.sh
# already queued, without restarting the pipeline.
#
# The stage's own attempts go through the HUC8s in order, one at a time.
# This submits side jobs that run the END of that list in parallel (whole
# HU4s, from HUC8 number --from on), then one last attempt of the stage that
# waits for the queued attempts and every side job, and points the next
# stage's job at that last attempt instead.
#
# Every job here gets RESUME=1, and with --resume the stage re-checks a HUC8's
# progress marker just before processing it, so whatever a side job finished
# is skipped by the attempts and loaded from its saved file. The last attempt
# fills in anything still missing (a side job that failed or timed out) and
# rewrites summary.csv over every HUC8 (each side job writes one covering
# only its own HU4s).
#
# Run on the login node (it only lists files and submits jobs):
#     bash data/slurm/parallel_huc8.sh --stage match_dams \
#         --after <last queued match_dams attempt> --next <first link_reaches attempt>
#
# Options:
#     --stage S     match_dams | link_reaches
#     --after J     the stage's last queued attempt (the job --next waits on)
#     --next J      the job that waits for the stage (afterok:J); it is
#                   pointed at the new last attempt
#     --from N      first HUC8 number (as in the log's [N/total]) the side jobs
#                   may take; HU4s that start before it are left to the
#                   attempts. Pick one the running attempt won't reach before
#                   its time limit. Default 220.
#     --jobs K      number of side jobs (default 8)
#     --wait J      side jobs start only after J succeeds (for link_reaches,
#                   the last match_dams attempt; not needed for a running stage)
#     --dry-run     print the plan, submit nothing

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
source data/slurm/case_study.env
case_study="$DAM_REMOVAL_CASE_STUDY"

stage="" after="" next="" from=220 jobs=8 wait_job="" dry=0
while (( $# )); do
    case "$1" in
        --stage)   stage="$2"; shift 2 ;;
        --after)   after="$2"; shift 2 ;;
        --next)    next="$2"; shift 2 ;;
        --from)    from="$2"; shift 2 ;;
        --jobs)    jobs="$2"; shift 2 ;;
        --wait)    wait_job="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$stage" || -z "$after" || -z "$next" ]]; then
    echo "--stage, --after and --next are required (see the top of this file)." >&2
    exit 2
fi

# The HUC8s in the order the stage processes them.
case "$stage" in
    match_dams)
        config="config/${case_study}.yaml"
        [[ -f "$config" ]] || { echo "No $config" >&2; exit 1; }
        huc8s=$(awk '/^huc8s:/ {f = 1; next}
                     f && /^[[:space:]]*-/ {gsub(/[^0-9]/, ""); print; next}
                     f && /^[^[:space:]#]/ {f = 0}' "$config") ;;
    link_reaches)
        dir="data/processed/station_reaches/${case_study}"
        huc8s=$(ls "$dir" 2>/dev/null | sed -n 's/^station_reaches_\([0-9]\{8\}\)\.parquet$/\1/p' | sort)
        if [[ -z "$huc8s" && -n "$wait_job" ]]; then
            echo "No station_reaches files yet in $dir; run this once select_station_reaches is done." >&2
            exit 1
        fi ;;
    *) echo "--stage must be match_dams or link_reaches" >&2; exit 2 ;;
esac
total=$(wc -w <<< "$huc8s" | tr -d " ")
(( total > 0 )) || { echo "No HUC8s found for $stage." >&2; exit 1; }

# Chunks of whole HU4s from HUC8 number $from on, about the same number of
# HUC8s each. One line per chunk: "<HUC8 count> <HU4> <HU4> ...".
chunks=$(tr ' ' '\n' <<< "$huc8s" | awk -v from="$from" -v jobs="$jobs" '
    NF { n++; hu4 = substr($1, 1, 4)
         if (!(hu4 in first)) { first[hu4] = n; order[++k] = hu4 }
         count[hu4]++ }
    END {
        for (i = 1; i <= k; i++) if (first[order[i]] >= from) { keep[++m] = order[i]; left += count[order[i]] }
        if (m == 0) exit
        size = 0; line = ""
        for (i = 1; i <= m; i++) {
            line = line " " keep[i]; size += count[keep[i]]
            # target: what is left spread over the chunks still to fill
            if (chunks < jobs - 1 && size >= left / (jobs - chunks)) {
                print size line; chunks++; left -= size; size = 0; line = ""
            }
        }
        if (line != "") print size line
    }')

echo "Stage $stage, case study $case_study: $total HUC8s."
if [[ -z "$chunks" ]]; then
    echo "No whole HU4 starts at or after HUC8 $from; nothing to run in parallel." >&2
    exit 1
fi
echo "Side jobs (HUC8s from number $from on, whole HU4s):"
awk '{printf "  %3d HUC8s: %s\n", $1, substr($0, index($0, " ") + 1)}' <<< "$chunks"
echo "Then a last $stage attempt after $after and the side jobs; $next will wait for it."
(( dry )) && { echo "(dry run: nothing submitted)"; exit 0; }

# Same as run_pipeline.sh: sbatch sometimes errors although the job WAS
# queued, so each submission carries a unique --comment and the queue is
# checked for it before retrying.
submit() {
    local tag="parallel_huc8_$$_$1" attempt id
    shift
    for attempt in 1 2 3 4 5; do
        if id=$(sbatch --parsable --comment="$tag" "$@" 2>/dev/null); then
            echo "${id%%;*}"; return 0
        fi
        sleep 10
        id=$(squeue -u "$USER" -h -o "%i %k" | awk -v t="$tag" '$2 == t {print $1; exit}')
        if [[ -n "$id" ]]; then echo "$id"; return 0; fi
        echo "  (sbatch failed for $tag; retrying)" >&2
    done
    echo "sbatch kept failing for $tag; stopping. Jobs above are queued." >&2
    return 1
}

side=()
wait_dep=()
[[ -n "$wait_job" ]] && wait_dep=(--dependency="afterok:$wait_job" --kill-on-invalid-dep=yes)
i=0
while read -r _ hu4s; do
    i=$((i + 1))
    export HU4="$hu4s" RESUME=1
    id=$(submit "side$i" --export=ALL --job-name="${stage}_side" \
         ${wait_dep[@]+"${wait_dep[@]}"} "data/slurm/${stage}.slurm")
    side+=("$id")
    echo "  side job $i: $id  (HU4 $hu4s)"
done <<< "$chunks"

deps="afterany:$after$(printf ':%s' "${side[@]}")"
unset HU4
export RESUME=1
last=$(submit last --export=ALL --dependency="$deps" \
       --kill-on-invalid-dep=yes "data/slurm/${stage}.slurm")
echo "  last attempt: $last  (after $deps)"

scontrol update JobId="$next" Dependency="afterok:$last"
echo "  $next now: $(scontrol show job "$next" | grep -o 'Dependency=[^ ]*')"
echo
echo "Watch: squeue -u \$USER; logs in data/slurm/logs/${stage}_<job>.out"
