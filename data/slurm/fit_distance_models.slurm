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
#
# Time since removal and dam / watershed characteristics (designs 11-12 of
# analysis/distance_models.py; added 2026-11-06). Not in the default RUNS:
# name them. They write to stage5/<run>__<RE>/, which the Stage 4 report
# doesn't read.
#   audit_10km, audit_30km
#                  no fits: time_coverage_by_year/_bin.csv (which time bins the
#                  data support) and modifier_audit.csv / modifier_correlations.csv
#                  (availability, missingness, collinearity). Run first; minutes.
#   time           binary BACI and before/after (the benchmark: the mixed_model.py
#                  formulas) next to the time-since-removal models, 10 km
#   time_ml        the same by maximum likelihood with intercepts: the
#                  likelihood-ratio test and AIC / BIC of time-varying vs constant
#                  effect (time_comparison.csv)
#   time_pretrend  BACI with the pre period split (late pre = reference): the early
#                  pre bin is a placebo for diverging pre-removal trends
#   time_ml_slopes SENSITIVITY for time_ml: the same ML comparison with dam random
#                  slopes in both models (intercepts-only likelihoods are
#                  overconfident where dams differ); the LRT / AIC is reported only
#                  where both fits have the same observations and random effects
#   time_coarse    SENSITIVITY for time: BACI with 1-2 / 3-5 year intervals instead
#                  of 1 / 2-3 / 4-5 (more dams per interval)
#   modifiers      each characteristic alone as a modifier of the effect, adjusted
#                  for distance and dilution, 30 km (modifier_screen.csv). 16 specs
#                  x every series: the longest run; LRT_BOOT=0 skips the bootstrap
#   modifiers_unadjusted   the same without the distance / dilution adjustment
#   modifiers_combined     several characteristics together: set COMBINED, e.g.
#                  COMBINED="dam_height pct_developed" RUNS=modifiers_combined sbatch ...
#                  Check combined_check.csv (written by the run) before reading it.
#
#   RUNS="audit_10km audit_30km" sbatch data/slurm/fit_distance_models.slurm
#   RUNS="time time_ml time_pretrend" sbatch data/slurm/fit_distance_models.slurm
#   RUNS="time_ml_slopes time_coarse" sbatch data/slurm/fit_distance_models.slurm
#   RUNS=modifiers sbatch data/slurm/fit_distance_models.slurm

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
    [audit_10km]="--stage stage5 --audit-only --max-km 10"
    [audit_30km]="--stage stage5 --audit-only"
    [time]="--stage stage5 --spec-sets time --max-km 10"
    [time_ml]="--stage stage5 --spec-sets time --max-km 10 --ml --re intercepts"
    [time_pretrend]="--stage stage5 --specs baci_binary baci_time --max-km 10 --time-bins pretrend"
    [time_ml_slopes]="--stage stage5 --analysis-role sensitivity --spec-sets time --max-km 10 --ml --re slopes"
    [time_coarse]="--stage stage5 --analysis-role sensitivity --specs baci_binary baci_time --max-km 10 --time-bins y1-2_y3-5"
    [modifiers]="--stage stage5 --spec-sets modifier_screen"
    [modifiers_unadjusted]="--stage stage5 --spec-sets modifier_screen_unadjusted"
    [modifiers_combined]="--stage stage5 --specs baci_x_combined before_after_x_combined --combined ${COMBINED:-}"
)
RUNS="${RUNS:-primary dilution size_proxy size_polygon dar_le_100 dar_le_10 dar_le_2 keep_suspects common_sample}"

# INPUT_VARIANT=min1yr (or min1yr_2mo): the same runs on
# analysis_measurement_<variant>.parquet, labelled --analysis-role sensitivity,
# into data/processed/mixed_model/<case_study>/sensitivity_<variant>/<run>__<RE>/
# (eligibility sensitivity analysis, docs/eligibility_sensitivity_prespec.md).
# Primary outputs (stage3/, stage5/) are not touched.
VARIANT_ARGS=""
if [[ -n "${INPUT_VARIANT:-}" ]]; then
    input="data/warehouse/$DAM_REMOVAL_CASE_STUDY/analysis_measurement_${INPUT_VARIANT}.parquet"
    [[ -f "$input" ]] || { echo "No $input (run build_analysis_dataset with ELIGIBILITY_VARIANT=$INPUT_VARIANT)"; exit 1; }
    VARIANT_ARGS="--input $input --analysis-role sensitivity --stage sensitivity_${INPUT_VARIANT}"
    echo "Sensitivity variant $INPUT_VARIANT: $input"
fi

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY; random effects: $RE; runs: $RUNS"
failed=()
for run in $RUNS; do
    if [[ -z "${ARGS[$run]+set}" ]]; then
        echo "Unknown run '$run'"; failed+=("$run"); continue
    fi
    # Runs that fix --re (it comes last and wins) are named for it.
    name="${run}__${RE}"
    if [[ "${ARGS[$run]}" =~ --re[[:space:]]+([a-z]+) ]]; then
        name="${run}__${BASH_REMATCH[1]}"
    fi
    if [[ "$run" == "modifiers_combined" && -z "${COMBINED:-}" ]]; then
        echo "modifiers_combined needs COMBINED=\"<characteristic> ...\""; failed+=("$run"); continue
    fi
    echo
    echo "=================== run: $name ==================="
    # ARGS is split into words on purpose.
    # shellcheck disable=SC2086
    python -u analysis/distance_models.py --run-name "$name" --re "$RE" \
        --lrt-bootstrap "${LRT_BOOT:-99}" ${ARGS[$run]} ${VARIANT_ARGS} || failed+=("$run")
done

echo
if (( ${#failed[@]} )); then
    echo "FAILED runs: ${failed[*]}"
    echo "Job completed with failures: $(date)"
    exit 1
fi
echo "Job completed: $(date)"
