# Pre-specification: BACI with controls known to be outside the former reservoir

**Written 2026-11-08, before any model was fit with this setting.** It fixes what is run, how it is compared with
the primary analysis and how the comparison is read. Changes after results are seen must be recorded at the end
with the reason.

## Why

BACI compares downstream stations with upstream control stations of the same dam. An upstream station that sat in
the former reservoir is itself changed by the removal, so it is not a valid control, and the models drop stations
flagged `upstream_in_former_reservoir`. That flag comes from a rough reservoir length (dam height ÷ river slope).
It is **unknown** when the dam has no height or slope, and the main analysis then treats the station as a control.

The former-reservoir audit (`notebooks/qa/reservoir_station_audit.ipynb`, 2026-11-08) found, in the main (3 / 2)
table:
- 47 of the 153 upstream control pairs used today (31%), at 38 dams, have **unknown** reservoir status;
- **38 dams rely only on unknown-status controls**, 21 of which also have downstream data (potential BACI dams);
- where a mapped reservoir polygon exists, **no** control station lies inside its dam's reservoir (reassuring, but
  polygons cover few dams).

So part of today's BACI evidence may rest on controls that were in the former reservoir. This sensitivity asks
whether the BACI results hold when only controls **positively known** to be outside it are used.

(The former reservoir as a study zone in its own right, reservoir before/after and three-zone BACI, was audited and
left out of the capstone: too few dams with controls above the reservoir, and an unreliable classification. It is
documented as a limitation and future direction.)

## What is run

The setting `controls_known_outside_reservoir = True` (`analysis/mixed_model.py` `assign_roles`; command-line flag
`--controls-known-outside` in `mixed_model.py` and `distance_models.py`) keeps an upstream station as a control only
when `upstream_in_former_reservoir` is **False**. Everything else is unchanged: treated stations, the 10 km / 30 km
windows, exposure rules, covariates, the suspect-dam exclusions, random-effects selection (`RE=auto`), SESOI,
minimum dams (10) and the exploratory label (< 15).

Runs, on the **main (3 / 2)** analysis table only, labelled `analysis_role = sensitivity`:
1. the mixed models (10 km): `KNOWN_CONTROLS=1 sbatch data/slurm/fit_mixed_model.slurm` →
   `mixed_model/<case>/sensitivity_known_controls/mixed_model/`;
2. the BACI distance model (`baci_distance`, 30 km, `RE=auto`): `RUNS=known_controls sbatch data/slurm/fit_distance_models.slurm`
   → `mixed_model/<case>/sensitivity_known_controls/known_controls__auto/`.

Before/after models have no controls and are not rerun. The primary outputs are never overwritten.

## Comparison and interpretation (fixed now)

Per BACI measure, with `analysis/sensitivity_compare.py` and the same labels and rules as the eligibility
sensitivity (`docs/eligibility_sensitivity_prespec.md`): the 10 km mixed-model effect, and the distance model's effect
at 5 km and distance slope.
- **robust / sharper:** the primary BACI result does not depend on unknown-status controls.
- **weaker / conflicting:** the primary BACI result is reported as sensitive to control validity.
- A model with fewer than 10 dams in either version is shown but not labelled ("not comparable"). Because this
  setting can only **remove** controls, many BACI measures may drop below 10 dams; that loss is itself reported
  (dams per measure, primary vs sensitivity) as a measure of how much BACI rests on unknown-status controls.
- No new measures, specifications or thresholds are added because of these results.

## Limitations

- "Known outside" relies on the same rough reservoir length; a station flagged False can still be misclassified.
- Dams with height and slope (the only ones whose controls can be confirmed) may differ from those without; the
  added/removed-dam comparison of `sensitivity_compare.compare_dams` is reported alongside.

## Changes after results were seen

*(none yet)*
