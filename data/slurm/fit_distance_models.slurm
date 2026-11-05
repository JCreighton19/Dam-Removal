#!/bin/bash
#SBATCH --job-name=fit_distance_models
#SBATCH --output=data/slurm/logs/distance_models_%j.out
#SBATCH --error=data/slurm/logs/distance_models_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Stage 3 distance / size models (analysis/distance_models.py), read-only on
# the warehouse. Each run writes results.csv, coefficients.csv,
# effects_at.csv and settings.json to
# data/processed/mixed_model/<case_study>/stage3/<run>/. The existing
# before_after / baci models (fit_mixed_model.slurm) are not touched.
#
#   sbatch data/slurm/fit_distance_models.slurm                  # every run below
#   RUNS="primary size_proxy" sbatch data/slurm/fit_distance_models.slurm
#
# Runs:
#   primary        BACI + before/after distance and scaled distance, treated
#                  rows within 30 km, log drainage-area ratio adjusting the
#                  effect, suspect dam matches excluded
#   dilution       distance x dilution interaction
#   size_proxy     reservoir_volume_proxy x effect, adjusted for log dam
#                  drainage area (exploratory)
#   size_polygon   the same with mapped reservoir area (mapped by the removal
#                  year only)
#   dar_le_100, dar_le_10, dar_le_2
#                  distance models on treated rows with ratio <= the limit
#   keep_suspects  distance models with the suspect dam matches kept in
#   primary_dam_slopes, size_dam_slopes
#                  the same with dam random slopes on the effect terms (falls
#                  back to intercepts if they don't fit; see re_structure):
#                  how much the intervals depend on assuming one effect for
#                  every dam

set -uo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

DISTANCE="baci_distance before_after_distance"
declare -A ARGS=(
    [primary]=""
    [dilution]="--specs baci_distance_dilution before_after_distance_dilution"
    [size_proxy]="--specs baci_size before_after_size"
    [size_polygon]="--specs baci_size before_after_size --size-variable polygon_pre_removal"
    [dar_le_100]="--specs $DISTANCE --max-dar 100"
    [dar_le_10]="--specs $DISTANCE --max-dar 10"
    [dar_le_2]="--specs $DISTANCE --max-dar 2"
    [keep_suspects]="--specs $DISTANCE --keep-suspect-dams"
    [primary_dam_slopes]="--specs $DISTANCE --dam-slopes"
    [size_dam_slopes]="--specs baci_size before_after_size --dam-slopes"
)
RUNS="${RUNS:-primary dilution size_proxy size_polygon dar_le_100 dar_le_10 dar_le_2 keep_suspects primary_dam_slopes size_dam_slopes}"

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; runs: $RUNS"
failed=()
for run in $RUNS; do
    if [[ -z "${ARGS[$run]+set}" ]]; then
        echo "Unknown run '$run'"; failed+=("$run"); continue
    fi
    echo
    echo "=================== run: $run ==================="
    # ARGS is split into words on purpose.
    # shellcheck disable=SC2086
    python -u analysis/distance_models.py --run-name "$run" ${ARGS[$run]} || failed+=("$run")
done

echo
if (( ${#failed[@]} )); then
    echo "FAILED runs: ${failed[*]}"
    echo "Job completed with failures: $(date)"
    exit 1
fi
echo "Job completed: $(date)"
