# Station QA before the refit

Two separate questions, two separate audits:

1. **Accuracy:** how reliable are the reach assignments of the 723 stations the analysis uses?
   (`analysis/audits/export_station_review.py`)
2. **Recovery:** how much useful data do the stations the matching did *not* accept hold?
   (`station_payoff_audit.py`, `station_payoff_recount.py`, `station_recovery_audit.py`)

Both are read-only. Nothing here changes the warehouse, the matching rules or the models. Corrections found
by either review go into `config/snap_overrides.csv` (`feature_kind = station`) only after the reviews are
finished, followed by one rerun from `select_station_reaches`, one model_freeze change-log entry and one refit.

## 0. Freeze the baseline (cluster, once, before anything changes)

```bash
B=data/processed/frozen/station-baseline-2026-11-16
mkdir -p $B
git rev-parse HEAD > $B/COMMIT
cp -a data/processed/station_reaches/all_removals $B/station_reaches
cp -a data/processed/dam_reaches/all_removals $B/dam_reaches
sha256sum data/warehouse/all_removals/analysis_measurement*.parquet \
    data/processed/reach_links/all_removals/reach_links_*.parquet > $B/INPUTS.sha256
```

After the accuracy export, also copy `data/processed/station_review/all_removals/station_review.csv` there:
it holds every analysis station's current decision, roles and risk tier.

## 1. Accuracy review of the analysis stations

`sbatch data/slurm/export_station_review.slurm` writes `station_review.gpkg` (layers `stations`,
`stations_snapped`, `snap_lines`, `chosen_reaches`, `flowlines` within 1 km, `dams`),
`station_review.csv` and `sample_design.csv`.

**Risk tier** (from the matching evidence; rules in the script's docstring):

| Tier | Meaning |
|---|---|
| manual | a manual override |
| high | without drainage-area or name support: a flagged conflict (confluence nearby, polygon edge, far snap, sink catchment) or an eligible rival within 25 m with a different name or very different drainage area; a name mismatch without drainage-area agreement; or snapped > 100 m from a point outside any water polygon |
| medium | any of those flags with drainage-area or name support; used although ambiguous (robust to the reach choice: same role either way); no drainage-area or name support; snapped > 50 m; a rival within 25 m |
| low | no flag |

Drainage-area or name agreement counts as resolving a flagged conflict because the conflicts are defined by a
rival reach at least 5x larger or smaller (a different river). Revised twice on 2026-11-16, before any
review: counting every flagged conflict as high put 80% of the stations there, and counting name-supported
ones 61%.

**Inspected individually** (`inspect_reason`): every BACI station (treated or control for a dam x measure
that has both), every manual and high-risk station.

**Random sample of the rest** (medium and low tier, not BACI): strata = tier x role (treated / control /
both / other) x feature type (StreamRiver / ArtificialPath) x drainage-area class (< 100, 100-1000,
>= 1000 km2). In each stratum max(3, 25%) of medium and max(3, 15%) of low stations, all if fewer, with a
fixed seed (`in_sample`, `sample_weight`). The error rate below is estimated separately per tier.

**Verdict** for every inspected or sampled station (`verdict` field): `correct`, `wrong_reach` (another
reach; give `correct_nhdplus_id`), `wrong_side` (right river, but on the other side of the dam, or in the
former reservoir), `not_a_stream` (a ditch, pond, well, outfall...), `unsure`. Review blind to results:
don't look at the station's water-quality data or effect estimates.

**Error rate of the strong assignments.** With N_h stations and n_h sampled in stratum h, e_h of them wrong
(`wrong_reach`, `wrong_side` or `not_a_stream`):

    p_hat = sum_h (N_h / N) * (e_h / n_h)
    var   = sum_h (N_h / N)^2 * (1 - n_h / N_h) * p_h (1 - p_h) / (n_h - 1)

Report p_hat with a 95% interval (normal, or Clopper-Pearson on the effective sample when no errors are
found; with 0 errors the one-sided 95% upper bound is about 3 / effective n). `unsure` counts as an error in
a sensitivity version. Expected undetected errors among all analysis stations =
(errors found in the inspected set) + p_hat x (low-tier, non-BACI stations not sampled).

**Decision rule** (fixed now, before reviewing): correct every error found. If the strong-assignment error
rate's upper bound is below 5%, no further review. If it is 5% or more, review the stratum (or strata) where
the errors occurred in full before the refit.

## 2. Recovery: unmatched stations

| Step | Script | What it gives |
|---|---|---|
| Payoff audit | `AUDIT=station_payoff sbatch data/slurm/stage1_audits.slurm` | structural upper bounds from links (done: 51 BACI / 15 BA dams) |
| Targeted recount | `sbatch data/slurm/station_payoff.slurm` | real counts for the 181 top stations (done: 4 BACI / 3 BA dams, 8 stations) |
| Structure | `STAGE=structure sbatch data/slurm/station_recovery.slurm` | every unmatched station (ambiguous and rejected) by reason, HUC2, distance, role, dam state |
| Summary | `STAGE=summary ...` | WQP summary counts (station x year x characteristic) for ambiguous stations linked within the limits; no result rows |
| Estimate | `STAGE=estimate ...` | per dam x station x parameter: would it qualify; gains in dams per parameter; priority list; calibration against the 181-station recount |

Gains, best first: `new_baci_dam` (a before/after-only dam gets a control), `new_ba_dam`, `new_baci_measure`
(a BACI dam gets a control for another measure it has downstream data for), `new_ba_measure`, `more_samples`
(more data for an existing dam x measure; precision only).

**Rejected stations** have no reach and no links, so their role is unknown. The structure stage reports how
many lie near a dated, accepted dam (straight line); they are a later step only if the ambiguous stations
show a worthwhile yield.

### Cutoff for review and targeted download

Fixed before seeing the estimates:

1. Only stations with an **eligibility gain** (the first four gains) are candidates. `more_samples` is not a
   reason to review on its own.
2. Order: `new_baci_dam`, `new_baci_measure`, `new_ba_dam`, `new_ba_measure`; within a gain, more dams, then
   more dam x measures. Stations confirmed by the 181-station download come first.
3. Use the calibration (precision of the summary-based prediction on the 181 downloaded stations) to turn
   estimated gains into expected real gains.
4. Review and download down the list while the **expected real gain is at least 1 dam (any measure) per 10
   stations reviewed**, or until the list ends. Below that, stop and report the remainder as a limitation.
5. Every recovered station still needs a QGIS check (provisional reach on the right river, right side of the
   dam, outside the former reservoir) before it becomes an override.

## 3. After both reviews

Corrections and recoveries go to `config/snap_overrides.csv` as `feature_kind = station` rows, together with
the dam corrections, then: rerun from `select_station_reaches` (`run_pipeline.sh --from select_station_reaches`),
compare against the frozen baseline (stations added / removed / moved; BA and BACI dams per measure), one
`docs/model_freeze.md` change-log entry, one refit snapshot.
