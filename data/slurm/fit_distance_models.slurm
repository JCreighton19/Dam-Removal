#!/bin/bash
#SBATCH --job-name=fit_distance_models
#SBATCH --output=data/slurm/logs/distance_models_%j.out
#SBATCH --error=data/slurm/logs/distance_models_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8

# Stage 3-4 distance / size models (analysis/distance_models.py), read-only on
# the warehouse. Each run writes results.csv, coefficients.csv,
# effects_at.csv and settings.json to
# data/processed/mixed_model/<case_study>/stage3/<run>__<RE>/. The existing
# before_after / baci models (fit_mixed_model.slurm) are not touched.
#
#   RE   random effects for every run: auto (default; dam slopes where the
#        data support them, tested with a boundary-correct likelihood-ratio
#        test: chi-square bound, else parametric bootstrap on the CPUs below),
#        intercepts, or slopes
#   RUNS which runs (default: all below)
#   LRT_BOOT  bootstrap draws for that test (default 99)
#
#   sbatch data/slurm/fit_distance_models.slurm                          # Stage 4 primary set
#   RE=intercepts RUNS="primary size_proxy" sbatch data/slurm/fit_distance_models.slurm
#   RE=slopes RUNS="primary size_proxy" sbatch data/slurm/fit_distance_models.slurm
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
#   common_sample  raw vs reservoir-length-scaled distance on exactly the same
#                  dams and days, maximum likelihood, intercepts for both (so
#                  AIC compares the distance terms, not random effects)

set -uo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

RE="${RE:-auto}"
DISTANCE="baci_distance before_after_distance"
ALL_DISTANCE="baci_distance baci_scaled_distance before_after_distance before_after_scaled_distance"
declare -A ARGS=(
    [primary]=""
    [dilution]="--specs baci_distance_dilution before_after_distance_dilution"
    [size_proxy]="--specs baci_size before_after_size"
    [size_polygon]="--specs baci_size before_after_size --size-variable polygon_pre_removal"
    [dar_le_100]="--specs $DISTANCE --max-dar 100"
    [dar_le_10]="--specs $DISTANCE --max-dar 10"
    [dar_le_2]="--specs $DISTANCE --max-dar 2"
    [keep_suspects]="--specs $DISTANCE --keep-suspect-dams"
    [common_sample]="--specs $ALL_DISTANCE --common-sample --ml --re intercepts"
)
RUNS="${RUNS:-primary dilution size_proxy size_polygon dar_le_100 dar_le_10 dar_le_2 keep_suspects common_sample}"

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; random effects: $RE; runs: $RUNS"
failed=()
for run in $RUNS; do
    if [[ -z "${ARGS[$run]+set}" ]]; then
        echo "Unknown run '$run'"; failed+=("$run"); continue
    fi
    # common_sample always uses intercepts (its --re comes last and wins).
    name="${run}__${RE}"
    [[ "$run" == "common_sample" ]] && name="${run}__intercepts"
    echo
    echo "=================== run: $name ==================="
    # ARGS is split into words on purpose.
    # shellcheck disable=SC2086
    python -u analysis/distance_models.py --run-name "$name" --re "$RE" \
        --lrt-bootstrap "${LRT_BOOT:-99}" ${ARGS[$run]} || failed+=("$run")
done

echo
if (( ${#failed[@]} )); then
    echo "FAILED runs: ${failed[*]}"
    echo "Job completed with failures: $(date)"
    exit 1
fi
echo "Job completed: $(date)"
