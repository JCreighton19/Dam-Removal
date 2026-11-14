# Manual dam review (2026-11-14)

**Why.** Evidence-based matching put some dams on the wrong reach, mostly through the reservoir-outlet rule
(side ponds next to the named river). Others can't be placed by any rule: e.g. CO-022, where the reported
location, the nearest reach and the named stream are all wrong. A misplaced dam puts every linked station on
the wrong side or at the wrong distance. So every dam that could contribute to the analysis is checked by hand
against imagery, and corrections are recorded in `config/snap_overrides.csv`. The dam data itself is never
edited.

**Rule.** Make and commit every geographic correction **before** looking at any revised statistical or predictive
result. The review corrects locations; it does not tune results.

## 1. Export (cluster)

```bash
sbatch data/slurm/export_dam_review.slurm
```

Writes `data/processed/dam_review/<case_study>/dam_review.gpkg` and `dam_review_dams.csv`
(`analysis/audits/export_dam_review.py`). The review set is dated dams only, one priority each:

| Rank | Priority | Dams |
|---|---|---|
| 1 | `flagged` | known suspects, the 2026-11-13 flags, and preprocessing_audit §13b "different path" / "path unknown" disagreements |
| 2 | `baci` | in the analysis table with an assigned downstream row and a valid upstream control |
| 3 | `analysis` | any other dam in the analysis table |
| 4 | `recoverable` | not in the analysis table, no accepted reach or no usable link, a WQP station within 30 km |

Dams with no station within 30 km are left out: no placement can make them usable. Dams without coordinates
in the dam list can't be exported.

Layers (EPSG:4326): `dams` (reported points, with `reviewed` / `review_note` for your own bookkeeping),
`dams_current` (where accepted dams sit now), `flowlines` (every NHDPlus flowline within 3 km of a dam's reported
or current point, from every package the circle touches), and an empty `corrections` layer.

The correct reach is farther than 3 km away? Export those dams with a wider circle to a separate file:

```bash
DAMS="CO-022 WI-043" RADIUS_KM=6 sbatch data/slurm/export_dam_review.slurm   # -> dam_review_extra.gpkg
```

Copy to the Mac: `scp <you>@<cluster>:<repo>/data/processed/dam_review/all_removals/dam_review.gpkg ~/Desktop/`

## 2. Review (QGIS, local)

1. Open the GeoPackage (all four layers). Add a satellite basemap: *XYZ Tiles → New connection*, Esri World
   Imagery `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`.
2. Label `flowlines` with `gnis_name` and `dams` with `AR_ID`. Filter or sort `dams` by `priority_rank`.
3. Snapping: *Project → Snapping Options*: enable, *Advanced configuration*, layer `flowlines`, type
   *Segment*, about 12 px.
4. Optional: in `corrections` → *Properties → Attributes Form*, set the default value of `nhdplus_id` to
   `array_first(overlay_nearest('flowlines', "nhdplus_id"))`, so it fills in the line you snapped to.
5. For each dam, in priority order:
   - **Correct:** nothing to record (tick `reviewed` in `dams` if you like).
   - **Wrong place:** toggle editing on `corrections`, add a point **snapped onto** the right flowline at the
     dam, and fill in `AR_ID`, `action = move`, `nhdplus_id` (the flowline's; check with *Identify*), `reason`
     and `source` (imagery, Street View, a news article, ...).
   - **Can't be identified with confidence** (e.g. the channel isn't in NHDPlus): a point anywhere near the
     dam with `action = reject`, no `nhdplus_id`, and the reason.
6. Save edits. You can stop after any priority group. Keep an untouched copy of the exported GeoPackage as the
   audit record.

## 3. Corrections → overrides (local)

```bash
python etl/apply_dam_review.py --review ~/Desktop/dam_review.gpkg --reviewer jc --dry-run
python etl/apply_dam_review.py --review ~/Desktop/dam_review.gpkg --reviewer jc
```

Checks every correction (AR_ID in the review, once only; valid action and reason; a move point within 25 m of
its flowline, which must be in the GeoPackage) and writes nothing if any fails. Each `move` becomes
`dam,<AR_ID>,accept,<nhdplus_id>,<reason [source]>,<reviewer>,<date>,<lat>,<lon>`; each `reject` an override
with no reach. Earlier overrides for the same dams are replaced; other rows are kept.

Then commit and push `config/snap_overrides.csv`, `git pull` on the cluster, and add a dated entry to the change
log in `docs/model_freeze.md` (scope reviewed, dams moved / rejected, reason). The frozen tag and outputs stay
as they are.

## 4. Rerun (cluster)

```bash
bash data/slurm/run_pipeline.sh --from match_dams
```

`match_dams.py` locates, assigns a HUC8 to, matches and places each corrected dam from its reviewed coordinates
(`coordinates_source = manual_correction`; `reported_latitude` / `_longitude` keep the dam list's values). It
fails a HUC8 if a corrected point is more than 50 m from its override reach, and stops if a corrected point lies
in no case-study HUC8. If a recovered dam's HUC8 isn't in the case study, add it to the config and rerun from
`download_wqp_stations` instead (its stations were never downloaded).

Then:
1. Rerun `notebooks/qa/dam_funnel.ipynb` and `preprocessing_audit` §13b; compare before and after: matched,
   recovered and rejected dams, changed reaches and HUC8s, BA and BACI dams, dams and observations per measure.
2. Refit the mixed, distance, Stage 4 and Stage 5 models, as a new versioned snapshot (record the code
   version), and rerun the results notebooks.

## 5. Afterwards (optional)

Fix the reservoir-outlet rule in `etl/common/dam_matching.py` (the outlet's own reach must match the dam's
River; a much smaller outlet loses to a nearby named river), for dams outside the review and future data.
