# Pre-specification: can pre-removal characteristics predict a dam's water-quality response?

**Written 2026-11-10, after the statistical models were frozen (`docs/model_freeze.md`) and before any predictive
model was fit or any dam-level target was computed.**

This is a **post-hoc, read-only** analysis of the frozen data. It reruns no ETL, refits no frozen model and changes
no frozen output. Stage 1 (the data audit) may refine *how* the target is built. Any such refinement is recorded,
with the audit numbers behind it, under "Amendments" at the end, **before** any predictive result is seen. The
decision criteria (section 8) are fixed now and are not changed after results are seen.

## 1. Question

> Can characteristics known before a dam is removed predict the size and direction of the subsequent water-quality
> change at its downstream monitoring stations?

- **Prediction is not causation.** A characteristic that predicts larger changes is associated with them across dams.
  It doesn't show that the characteristic causes them, or that the removal caused the change (observational data,
  before/after design).
- **A negative result is an acceptable, reportable outcome:** *"site-level water-quality responses following dam
  removal were not reliably predictable from the available pre-removal dam and watershed characteristics."*
- **What the frozen results lead us to expect:**
  - dams differ a lot (dam-to-dam SD above the threshold for 15 of 16 distance models)
  - the characteristic screen found few associations beyond chance in before/after: 7 of 79 tests had an interval
    excluding 0, against about 4 expected, despite high power

  So modest or no predictive skill is the expected result. The analysis is designed to find that out cheaply and
  honestly.

## 2. What the frozen pipeline does (inspected 2026-11-10; the code is the source of truth)

| Item | Implementation |
|---|---|
| Analysis table | `data/warehouse/<case>/analysis_measurement.parquet`: one row per measurement × eligible dam-station pair (`etl/build_analysis_dataset.py`) |
| Window and `post` | config `wq_window`: 5 years before and 5 after. `years_from_removal` = sample year − removal year. `period` = `pre` for −5…−1 and `post` for +1…+5; the **removal year (0) and years beyond ±5 are not in the table**. `post = 1` when `period == "post"` (`mixed_model.prepare`) |
| Removal year | `removal_year` (integer year; no month or day) |
| Downstream / treated | `side == "downstream"` and `exposure_status` in `assigned_sole`, `assigned_nearest`, `assigned_overlapping` (`mixed_model.assign_roles`) |
| Network distance | `network_distance_km`: river distance dam → station along the NHDPlus network, computed in a projected CRS during linking. The warehouse window is 30 km; the 10 km mixed models keep treated rows with distance ≤ 10, and rows with no distance are dropped |
| Former reservoir | `upstream_in_former_reservoir`: an upstream station within the rough reservoir length (height ÷ slope). It concerns controls only, so it doesn't affect before/after |
| Row filters | `qc_non_perennial` rows dropped. SSC and TSS kept only on days with discharge measured. One value per dam × station × series × day (the median). Log10 for skewed measures (floor 0.001) |
| Season and weather | `C(month)` + z-scored `air_temp_mean_c` (sample-month air temperature) and `precip_prev_month_mm` (previous-month precipitation) |
| Frozen before/after model (10 km) | `y ~ post + C(month) + z_air_temp_mean_c + z_precip_prev_month_mm`, random intercept and **random slope on `post` by dam**, station intercepts, t intervals with dams − 1 df. `results.csv` column `effect_sd_between_dams` = SD of the dam random slope |
| `dam_effects.csv` | `data/processed/mixed_model/<case>/dam_effects.csv`: `AR_ID, dam_effect, days, stations, series, model`. `dam_effect` = fixed `post` + the dam's random slope, a **BLUP shrunk toward the average** |
| Distance models (primary results) | 30 km, `ld = ln(1+km) − ln(6)` and dilution terms, `RE=auto`, 12 suspect dams excluded (`config/suspect_dam_matches.csv`) |
| Region | Not in the analysis table. `HUC8` is on the warehouse `dam_removal` table; HUC2 and HUC4 are its first 2 and 4 digits |
| Conventions | scripts in `analysis/`, outputs under `data/processed/`, Slurm jobs in `data/slurm/` (sourcing `case_study.env`, logs in `data/slurm/logs/`), tests in `tests/` runnable as `python tests/test_x.py`, notebooks that only read outputs |

## 3. Differences between the original proposal and the data (resolved here)

1. **Post-removal periods.** The proposal's "year 0–1" and "> 5 years" don't exist in the table: the removal year
   is excluded and the window ends at +5. Adding them would need an ETL rebuild, which is out of scope. The available
   periods are **year 1, years 2–3 and years 4–5** (the frozen time models' bins). Removal dates are known only to
   the year, so "year 1" spans about 1–24 months after removal.
2. **Sample size at 10 km.** The ~100–280 dams per measure quoted earlier are the 30 km distance models. The frozen
   10 km before/after models used **66–134 dams** for the main candidate measures (turbidity 66, total P 80, TDS 47).
   The audit reports the exact counts under each window. Whether a distance-standardized 30 km target is defensible
   is decided at checkpoint 3, from the distance audit.
3. **Suspect dams.** The frozen 10 km model kept the 12 suspect dams; the distance models excluded them. The
   predictive analysis **excludes** them: a dam on the wrong stream gives a meaningless target.
4. **Climate.** The model's weather covariates are sample-month weather, including after removal, so they **can't
   be predictors**. The warehouse has no climate normals. The only pre-removal climate summary available is the mean
   over the dam's **pre-removal sample days**, which is weighted by when sampling happened. It is a candidate
   predictor with that caveat.
5. **Land cover** is per downstream station's watershed (fixed baseline: the last NLCD year before removal, else the
   nearest year). Dam-level value = the median over the dam's treated stations. It is used **only when
   `land_cover_year < removal_year`**; otherwise it's set missing, as a leakage guard for removals before 1986.
6. **Mapped reservoir area** is used only when `reservoir_polygon_timing == "edited_by_removal_year"` (as in
   `distance_models.CHARACTERISTICS`).
7. **Which variance to use for reliability.** The frozen 10 km dam random-slope variance (τ²) describes the same
   kind of quantity as a 10 km dam-average change. It is not identical, because the frozen model pools month and
   weather effects across dams. The distance models' random-slope variance is the effect at 5 km, a different
   estimand, and is not used for a 10 km target. A second, target-based τ² (section 6) is reported alongside.

## 4. Scope

- **Primary:** before/after (downstream) responses only. One prediction unit = **one dam × measure**. Stations,
  days and distance or time bins are never separate training examples.
- **BACI is excluded** from prediction: 12–30 dams per measure is too few for geographic validation. It is a
  possible future exploratory extension.
- **Candidate measures:** water temperature, dissolved oxygen, pH, specific conductance, total phosphorus,
  turbidity, TDS. SSC, TSS and orthophosphate are audited but go forward only if they meet the rule below.

**Rule for a measure to go to modeling** (applied to the Stage 2 targets):
- at least **60 dams** with an estimable target (about 15 dams per predictor at 4 predictors), and
- at least **5 HUC2 regions with ≥ 5 dams each** (so leave-one-region-out validation has real folds), and
- target reliability **≥ 0.15** (section 6).

## 5. Candidate predictors (pre-removal only)

Dam-level, in this priority order (used to break ties when two are collinear):

1. log10 dam height (`dam_height_ft`)
2. log10 dam drainage area (`dam_drainage_area_km2`)
3. log10 reservoir-volume proxy (`reservoir_volume_proxy`; a constructed index built from height, slope and drainage
   area, so likely collinear with 1–2)
4. % agriculture, 5. % forest, 6. % developed (median over the dam's treated stations; see 3.5)
7. pre-removal mean air temperature and 8. pre-removal mean precipitation (pre-period sample days; see 3.4)
9. dam age at removal (`dam_age_at_removal`)
10. log10 mapped reservoir area (pre-removal only; expected to be sparse)

Used only in baselines or sensitivity analyses, never in the primary ridge model:
- **HUC2 region:** model B, the region-only baseline.
- **Removal year:** a sensitivity check. It mixes calendar-era effects such as monitoring practice and providers
  with any real trend.
- **Latitude / longitude:** not used. Region is handled by the baseline and by the validation grouping.

Never used: anything measured after removal, post-removal weather, anything computed from the outcome, `original_use`
(free text, too many levels for these sample sizes), station distance (a sampling feature, not a site feature; it
is handled in the target).

**Selection rules (fixed now, applied to the Stage 1 audit, never to model performance):**
- Drop a predictor with < 70% completeness among the estimable dams of a measure.
- For a pair with |Spearman ρ| > 0.8, keep the higher-priority one. Report variance inflation factors (VIF) for
  what remains.
- At most ⌊n_dams / 15⌋ predictors per measure, taken in priority order.
- Remaining missing values: median imputation plus a missing-indicator column, **fitted inside each training fold**.
  Complete cases only is a sensitivity analysis.

## 6. Target (Stage 2; finalised at checkpoint 3)

**Conceptual estimand.** For dam *i* and measure *m*, the adjusted before/after change at its downstream stations,
averaged over years 1–5 after removal, on the frozen model's scale (log10 for skewed measures), on a distance basis
that is comparable across dams.

**Requirements:**
- **Estimated independently for each dam:** a separate fit per dam, not the BLUPs. The BLUPs (`dam_effects.csv`) are
  a sensitivity target only.
- **Adjusted for season and weather** as in the frozen model (month, air temperature, previous-month precipitation),
  with station-specific baselines.
- **Each dam counts once,** however many stations it has. Stations within a dam are combined so that one heavily
  sampled station doesn't dominate.
- **Distance:** the preferred estimand is a distance-standardized average Δᵢ = ∫ Δᵢ(d) w(d) dd with a common w(d).
  It is adopted only if the Stage 1 distance audit shows enough dams have stations across a common range. Otherwise
  a narrower common-support range, or a dam-level 0–10 km average with its station distances reported, is chosen
  and justified at checkpoint 3.
- **Time:** the primary target is the average over the available post years. Event-time responses (year 1, years
  2–3, years 4–5) are secondary and estimated only where the audit shows coverage. They are never extra training
  rows.
- **Standard error:** must allow for day-to-day autocorrelation (e.g. years as the unit of replication). The method
  is fixed at checkpoint 3, before reliability or any model is computed.
- **Provisional estimability:** at least one station with pre- and post-removal days, and samples in ≥ 2 distinct
  years on each side. The audit reports counts under 1, 2 and 3 years so the final rule is evidence-based.

**Sensitivity targets:** the frozen BLUPs, alternative distance weightings, and alternative time definitions where
supported. The analysis reports how much the targets change (correlation and mean absolute difference).

**Reliability (Stage 2B):** R = τ² / (τ² + mean SEᵢ²), using:
- (a) τ² from the frozen 10 km before/after random slope (see 3.7), and
- (b) τ² estimated from the targets themselves, as max(0, var(Δ̂ᵢ) − mean SEᵢ²).

R is an approximate guide to how much of the observed dam-to-dam variation is real rather than measurement noise,
**not a strict upper bound** on the cross-validated R². Below 0.15 under both (a) and (b), the measure stops before
modeling.

## 7. Validation and models (Stage 3)

**Validation:**
- **Primary:** leave-one-HUC2-region-out, with each region's dams held out together. Regions with < 5 dams for a
  measure are merged into one "small regions" fold.
- **Sensitivity:** 5-fold cross-validation grouped by HUC4, repeated 20 times with different group assignments.
- **No random splits** of dams, and no stations of one dam split across folds.
- **All preprocessing is fitted on training folds only:** imputation, scaling, predictor selection and penalty
  choice.

**Models:**
- **A, overall mean:** predicts the training-fold mean of the target for every test dam.
- **B, region only:** predicts the training-fold mean of the dam's HUC2 region. A region absent from training falls
  back to the overall mean, so leave-one-region-out makes B equal A by construction. B is therefore assessed under
  the HUC4-grouped scheme, where the region is present in training.
- **C, ridge regression (primary):** the section 5 predictors, standardized. The penalty is chosen by an inner
  grouped cross-validation over a fixed grid (10⁻²…10³), inside the training fold.
- **D, small tree ensemble (nonlinearity check):** random forest with 300 trees, maximum depth 3 and at least 10
  dams per leaf. No tuning.

Each dam has equal weight. Weighting by 1 / (τ² + SEᵢ²) is a sensitivity check. Seeds are fixed and recorded.

**Metrics,** on the pooled out-of-sample predictions, in natural units (°C, mg/L, pH, % for log measures):
- out-of-sample R², = 1 − SSE / SS around the mean of the training-fold means
- MAE and RMSE
- direction accuracy (sign of prediction = sign of target)
- per-fold results
- improvement over A and over B

AUC for "change beyond the threshold" only if both classes have at least 15 dams.

**Uncertainty:** a percentile interval from resampling **HUC2 regions** with replacement (cluster bootstrap of the
out-of-sample predictions, 2,000 draws). With few regions this interval is approximate, and the report says so.

## 8. Go / no-go per measure (fixed now)

A measure gets a final predictive model only if model C (or D, reported as exploratory) meets **all** of the
following:

1. Out-of-sample R² **> 0 and ≥ 0.05** under the primary validation.
2. It beats the region-only baseline B under HUC4-grouped validation (higher R²).
3. **Consistency:** it beats A in at least half of the folds with ≥ 10 dams.
4. **Stability:** the cluster-bootstrap 90% interval for R² excludes 0.
5. **Practical:** MAE at least 5% lower than model A's, with the MAE reported in natural units next to the
   threshold.

Otherwise: **no-go**, reported with the reliability, the validation results and the baselines. With 7 candidate
measures, one measure passing alone is reported cautiously (multiple comparisons).

## 9. Outputs and reproducibility

- **Code:** `analysis/predictive_feasibility.py` (stages `audit`, `targets`, `cv`), with tests in
  `tests/test_predictive_feasibility.py` (simulated data).
- **Job:** `data/slurm/predictive_feasibility.slurm`.
- **Outputs:** `data/processed/predictive/<case>/`, as machine-readable CSVs plus `report_<stage>.md`.
- **Notebook:** `notebooks/analysis/predictive_feasibility.ipynb`, which reads the outputs and fits nothing.
- **For Tableau:**
  - always: observed dam-level changes, with uncertainty and characteristics
  - only for measures that pass: predictions, labelled clearly as predicted

## Amendments (before any predictive result; with the audit evidence)

### Amendment 1: 2026-11-10, after the Stage 1 audit, before any target or model result

**Audit evidence** (`data/processed/predictive/all_removals/report_audit.md`, job of 2026-11-10):

- **Measures meeting the size rule** (10 km window, suspect dams excluded): water temperature 130 dams, pH 128,
  DO 124, specific conductance 109, total phosphorus 77, turbidity 64. Each has 7–8 HUC2 regions with ≥ 5 dams.
  - TDS (44 dams; 2 regions), TSS (30), orthophosphate (32) and SSC (7) **do not go forward**.
  - These counts match the frozen 10 km model's after removing the 4 suspect dams (e.g. 134 → 130).
- **The provisional ≥ 2 years each side rule removes no dam**: the 3 / 2 eligibility already guarantees it.
- **Distance:**
  - 74% of dams have a single station within 10 km.
  - Only 10–13 dams have stations ≥ 3 km apart, and 2 dams cover all of the 0–2, 2–5 and 5–10 km bands.
  - Station distances are spread across 0–10 km (median ≈ 4.6–5 km) and differ by region (median 1.8 km in
    HUC2 05 vs ≈ 7 km in 01 and 03).
  - **Dam-specific distance responses are not estimable.** The ∫Δᵢ(d)w(d)dd estimand of section 6 is not supported.
- **Time:** ≈ 95% of dams have year 1 and years 2–3 data, ≈ 73% have years 4–5, and ≈ 70% have all three. Median
  ≈ 66–80 daily values per dam over 10 years. **Dam-specific time profiles are not estimable.**
- **Predictors:**
  - dam age (≈ 38% complete) and mapped reservoir area (≈ 20%) fail the 70% rule
  - dam height ≈ 80% and the volume proxy ≈ 73% complete
  - no pair has |ρ| > 0.8; the volume proxy has the highest VIF (≈ 6)

**Decisions** (approved by the analyst 2026-11-10):

1. **Measures:** water temperature, DO, pH, specific conductance, total phosphorus, turbidity.
2. **Primary estimand:** *the estimated average downstream water-quality change associated with a dam removal,
   standardized to a common 5 km network distance and reference dilution condition, over the available
   post-removal observation period (years 1–5).* One dam per measure is one prediction unit. No dam-specific
   distance or time curves are estimated.
3. **Distance:** standardized to 5 km and the reference dilution with the **frozen pooled before/after
   distance-model coefficients** (`stage3/primary__auto`, spec `before_after_distance`, not refit).
4. **Window:** 10 km primary, 30 km as a separate sensitivity analysis.
5. **Predictors:** the reservoir-volume proxy is dropped from the primary set. It is a function of height, slope
   and drainage area, has the highest VIF and 27% is missing. Missing-value indicators count toward the
   15-dams-per-predictor cap, and are added only for predictors with ≥ 10% missing among a measure's dams.
6. **Time:** years 1–5 primary, years 1–3 as a sensitivity target.

**Target definition.** Per measure, one fixed-effects regression on the frozen before/after daily rows (10 km,
suspect dams out):

  y_{s,t} = α_s + Σ_m γ_m·1[month = m] + θ₁·z_air + θ₂·z_precip + δ_s·post_{s,t} + ε_{s,t}

- α_s is a baseline for each station s, and δ_s is that station's own before/after change (not pooled across
  stations or dams).
- Month and weather effects are pooled across dams, as in the frozen model. The covariates are z-scored within
  the fit.
- The dam target averages its stations with equal weight, and standardizes with the frozen coefficients b_ld,
  b_ldar (the `post_ld`, `post_ldar` terms):

  Δᵢ = (1/Sᵢ) Σ_{s∈i} δ_s − b_ld·l̄dᵢ − b_ldar·l̄darᵢ

  where l̄dᵢ is the mean over the dam's stations of ld = ln(1 + km) − ln(6), and l̄darᵢ is the mean of
  ldar = log10(ratio) − log10(reference ratio of the frozen fit).
- Units are the frozen model's (log10 for conductance, phosphorus and turbidity).
- A station with no drainage-area ratio can't be standardized. It is left out of the standardized target (and
  counted), but kept in the unstandardized sensitivity target.

**Uncertainty.**
- **Sampling error:** cluster-robust with **station × sample-year clusters**, using **CR2** (Bell–McCaffrey) as
  primary, with CR1 reported.
  - Dam SE² = (1/Sᵢ²)·1ᵀ V_{ii} 1, where V_{ii} is the block of the sandwich covariance for the dam's δ_s.
  - 95% CI: t with Gᵢ − 1 df, where Gᵢ is the number of station-year clusters for the dam. This is an approximation.
  - Reported per dam: the number of clusters, days and stations.
  - **Robustness check:** a station-year-means estimate and SE. Weather- and month-adjusted values are averaged per
    station-year; a station's change is the mean of its post-year means minus the mean of its pre-year means, with
    a Welch SE across years. This is a check, not a replacement for CR2.
- **Uncertainty in the frozen coefficients:** the SEs of b_ld and b_ldar are saved, but **their covariance is not**,
  and the models are not refit.
  - Delta method: Var_coef,i = l̄dᵢ²·se_ld² + l̄darᵢ²·se_ldar² + 2·l̄dᵢ·l̄darᵢ·cov.
  - The unknown covariance is bounded by |cov| ≤ se_ld·se_ldar. The **primary total SE uses the conservative bound**
    (l̄dᵢ·se_ld and l̄darᵢ·se_ldar added in absolute value); the ρ = 0 version is also reported.
  - The conditional SE (given the coefficients) is reported separately. The coefficient error is **shared across
    dams** (it scales with each dam's distance and dilution), so adding it to each dam's SE treats a correlated
    error as independent. That is conservative for reliability.
- **Weak uncertainty support** is flagged for a dam with any of:
  - fewer than 6 station-year clusters
  - fewer than 20 daily values
  - conditional SE above 3× the measure's median
  - year-means SE ÷ CR2 SE outside 0.5–2

**Reliability.** R = τ² / (τ² + mean SEᵢ²), plus the mean of the per-dam τ² / (τ² + SEᵢ²). τ² comes from:
- **(a) frozen 10 km mixed model:** `effect_sd_between_dams`² (an SD, squared). This is the between-dam variance of
  the *unstandardized* ≤ 10 km change, so it's an approximation for the standardized target.
- **(b) frozen 30 km distance model:** `var_dam_post` (already a variance). It is the between-dam variance of the
  change *at 5 km and the reference dilution*, the same estimand as Δᵢ, but estimated from the 0–30 km sample.
  **Used for the 30 km sensitivity target;** reported alongside (a) for the primary target.
- **(c) from the targets themselves, accounting for measurement error:** a random-effects meta-analysis of the
  Δ̂ᵢ with their SEᵢ, using Paule–Mandel (primary) and DerSimonian–Laird. This estimates the between-dam variance
  left after removing each estimate's own sampling variance. It is **not** the raw variance of Δ̂ᵢ.

R is computed with the total SE (primary, conservative) and with the conditional SE. The section 4 stop rule
(< 0.15) applies to (c) with the total SE, and to (a) or (b). A measure stops only if all of them are below 0.15.

**Sensitivity targets** (compared with the primary: dams, Pearson and Spearman correlation, mean absolute
difference):
1. unstandardized
2. nearest station only
3. years 1–3
4. the frozen 10 km BLUPs (`dam_effects.csv`)
5. 30 km window: the same regression and standardization, all treated stations ≤ 30 km. The added dams and
   regions are reported.

**Stage 2 stops at checkpoint 3.** No predictive model is fit until the targets, their uncertainty, the reliability
and the sensitivity results have been reviewed.

### Amendment 2: 2026-11-11, after checkpoint 3, before any predictive model is fit

**Evidence** (`report_targets.md`, job 11815004 of 2026-11-11):

- **Size:** dam counts match the audit (130, 128, 124, 109, 77, 64), and no station lacked a dilution ratio.
- **Reliability:** 0.29–0.78 for every τ² source. The lowest is water temperature (0.29–0.44), the highest
  phosphorus (0.62–0.78). **No measure meets the stop rule.**
- **Uncertainty:** the year-means and CR2 SEs agree (median ratio 1.01), and the year-means targets correlate
  0.90–0.97 with the primary. These checks support the uncertainty estimates, but they don't prove that every
  source of correlated error is captured.
  - Weak support is flagged for 14–29 dams per measure (≈ 15–25%).
- **Standardization:** it barely matters; the unstandardized targets correlate 0.99–1.00 with the primary.
- **Agreement with the other target versions:**
  - nearest station, frozen BLUPs and 30 km mostly r ≈ 0.93–0.98
  - specific conductance: Pearson 0.73 (nearest station) and 0.82 (BLUPs) against Spearman 0.97–0.98, so a few
    dams or stations drive that disagreement
  - years 1–3 vs 1–5: r = 0.66 for total phosphorus and 0.73 for turbidity (≈ 0.93 for the others)

**Decisions** (approved by the analyst 2026-11-11):

1. **Checkpoint 3 is approved.** All six measures go to Stage 3 with the Amendment 1 primary targets unchanged.
2. **Three sensitivity analyses are added.** The primary analysis stays the full sample.
   - **(a) Weak support excluded:** the Stage 3 analysis is repeated without the dams flagged for weak uncertainty
     support.
   - **(b) Influence of the conductance outliers:**
     - The dams that drive the Pearson/Spearman gap are listed: the change in Pearson r when each dam is left out,
       against the nearest-station and BLUP versions.
     - A leave-one-dam-out influence check on the primary cross-validated R² (models A and C).
     - **No dam is removed from the primary analysis** unless a real data-quality problem is found. Such a removal
       would be recorded as a separate amendment, with its reason, before any result that depends on it is used.
   - **(c) Timing:** the Stage 3 analysis is repeated on the years 1–3 targets. It is run for all measures, but the
     **prespecified comparison is total phosphorus and turbidity.** The primary prediction concerns the *average
     response over years 1–5*, not a fixed or immediate post-removal response, and is reported that way.
3. **Stage 3 implementation details,** fixed now so that §5 and §7 are unambiguous:
   - **Predictor selection is inside each training fold,** using training features only and never the target.
     - The candidates are §5 without the volume proxy (Amendment 1.5), in priority order.
     - A predictor needs ≥ 70% completeness, and it is skipped if |Spearman ρ| > 0.8 with one already chosen.
     - A missing-value indicator is added for a predictor with ≥ 10% missing among the training dams.
     - Each predictor costs 1, plus 1 for its indicator, against the cap ⌊n_train / 15⌋. Selection stops at the
       first predictor that doesn't fit; it doesn't skip ahead to a lower-priority one.
     - The full-sample selection is also reported, as the set a final model would use, along with how often each
       predictor was chosen across folds.
   - **Ridge penalty:**
     - 11 log-spaced values from 10⁻² to 10³; ties go to the larger penalty. The intercept is not penalized.
     - Imputation and scaling are refitted in every inner fold.
     - Inner folds: leave-one-HUC2-out (small regions merged) under the primary scheme, and grouped 5-fold by HUC4
       under the HUC4 scheme. If fewer than 2 inner groups exist, grouped 5-fold by HUC4 is used.
   - **HUC4 folds:** the HUC4 groups are shuffled and each is assigned to the fold with the fewest dams so far.
     There are 20 repeats, and the seed is 20261111 (recorded in the outputs).
   - **Go/no-go criteria:**
     - **2:** R² averaged over the 20 HUC4 repeats, C against B.
     - **3:** "beats A" means a lower mean squared error in that leave-one-region-out fold. Only folds with
       ≥ 10 dams count.
     - **4:** a cluster bootstrap of the leave-one-region-out predictions over the HUC2 regions, 2,000 draws. The
       5th percentile of R² must be > 0.
     - **5:** leave-one-region-out MAE. For the log10 measures, MAE in natural units is reported as (10^MAE − 1)
       × 100%.
     - Model D's go flags are reported as **exploratory.**
   - **The other prespecified sensitivity analyses (§5, §7):**
     - **Weighted:** weights 1 / (τ² + SEᵢ²), with τ² by Paule–Mandel on the training fold's targets (total SE),
       scaled to mean 1. The weights apply only to fitting; the evaluation is unweighted.
     - **Complete cases:** dams complete on the full-sample selected predictors.
     - **Removal year:** added to the selected set, outside the cap.
     - **30 km window:** the 30 km targets, with the land-cover medians and pre-removal climate taken over the
       treated stations within 30 km.
   - **AUC:** the class is |target| > the SESOI (`stage4_report.SESOI`) and the score is |prediction|. It is
     reported only if both classes have ≥ 15 dams.
   - **The sensitivity analyses don't change the go/no-go decision.** Their results are reported next to it.

**Stage 3 is built and tested on simulated data only.** It runs on the cluster only after the code has been
reviewed.
