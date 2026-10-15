"""
Pilot eligibility report: for the dam-station links produced by
link_dams_stations.py, determine how many dams (and how many individual
dam-station pairs) have usable water-quality observations in the
pre-removal window, the post-removal window, or both -- overall and broken
down by WQP CharacteristicName.

This exists to validate analytical feasibility before scaling the pipeline
beyond the pilot watershed: if very few dams end up with comparable
before/after records, that is a finding to act on now, not after building
out national infrastructure.

Temporal window definition
---------------------------
- `removal_year` comes directly from the dam-station link table's
  `year_removed` column (sourced from the dam dataset's Year_Removed
  field). Only the year of removal is recorded, not an exact date.
- pre-removal window  = the `years_before` calendar years strictly BEFORE
  removal_year: [removal_year - years_before, removal_year - 1].
- post-removal window = the `years_after` calendar years strictly AFTER
  removal_year: [removal_year + 1, removal_year + years_after].
- The removal year itself is EXCLUDED from both windows. Because only the
  year (not the date) of removal is known, a measurement recorded during
  the removal year could have occurred before or after the actual removal
  date -- it is treated as ambiguous rather than guessed into either side.
- `years_before`/`years_after` come from the case-study config's
  `wq_window`, so the window width is a single documented setting, not a
  hardcoded number.

Removal-year and relationship handling
---------------------------------------
- Dam-station links with a missing, non-numeric, or zero `year_removed`
  are EXCLUDED from eligibility entirely and reported separately as
  "dams_excluded_missing_removal_year" / "excluded_missing_year" -- they
  are never folded into a "zero eligible observations" count, since that
  would conflate "we don't know" with "we checked and there's nothing".
- Only links already classified by link_dams_stations.py as "upstream",
  "downstream", or "same_reach" (i.e. network-connected) are considered.
  "unconnected" pairs are excluded the same way "no removal year" pairs
  are: dropped from the eligibility universe, not counted as ineligible.

Dam-level vs. pair-level eligibility
-------------------------------------
- A dam-station PAIR counts as having "both" only if that single station
  has qualifying observations in both the pre- and post-removal windows.
- A DAM counts as having "both" only if at least one of its linked pairs
  individually has "both" (i.e. dam-level "both" is "any pair has both",
  NOT "any pair has pre" AND "any (possibly different) pair has post").
  Two different stations independently covering the pre- and post-periods
  do NOT make a dam eligible for a before/after comparison.
- "has_pre" and "has_post" at the dam level ARE allowed to be satisfied by
  different pairs (that's a weaker, descriptive statistic, not the
  before/after eligibility claim) -- which is why both the looser
  has_pre/has_post counts and the stricter has_both count are reported
  side by side; read has_both as the one that matters for "can we do a
  before/after comparison for this dam".

Characteristics
----------------
- `CharacteristicName` values are taken verbatim from the WQP source data
  and are never merged or treated as interchangeable (e.g. "Oxygen" and
  "Dissolved oxygen (DO)" are different characteristics here, because they
  are not known to share units or measurement method). The
  per-characteristic breakdown applies the exact same same-pair "both"
  rule independently within each characteristic.
- The overall has_both of this script's does not require the SAME
  characteristic in both periods. The per-characteristic report in
  `pilot_eligibility_by_characteristic_{huc8}.csv` answers that stricter
  question.

Missing inputs vs. zero eligible observations
-----------------------------------------------
Required input files are checked up front. If the dam-station link table
or the WQP result batches are missing, the manifest records
status="missing_input" (with the specific missing path) and the script
exits non-zero -- this is never conflated with a successful run that
genuinely finds zero eligible dams/pairs (status="ok" with explicit 0
counts).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common.config import default_case_study, load_config
from etl.common.io import require_columns

CONNECTED_RELATIONSHIPS = ("upstream", "downstream", "same_reach")

LINKS_REQUIRED_COLUMNS = {"dam_id", "dam_name", "station_id", "year_removed", "relationship"}
RESULTS_REQUIRED_COLUMNS = {"MonitoringLocationIdentifier", "CharacteristicName", "ActivityStartDate"}

MANIFEST_FIELDS = [
    "status",
    "huc8",
    "years_before",
    "years_after",
    "removal_year_handling",
    "characteristics",
    "dams_total_linked",
    "dams_excluded_missing_removal_year",
    "pairs_considered",
    "dams_eligible_pre",
    "dams_eligible_post",
    "dams_eligible_both",
    "pairs_eligible_pre",
    "pairs_eligible_post",
    "pairs_eligible_both",
    "generated_at",
    "error_message",
]


# --------------------------------------------------------------------------
# Pure logic (no file I/O) -- exercised directly by tests/test_eligibility_window.py
# --------------------------------------------------------------------------

def prepare_links(links):
    """Resolve the universe of (dam, station) pairs eligible for consideration.

    Returns (pairs, excluded_missing_year) where `pairs` has one row per
    unique (dam_id, station_id) with a usable removal_year, and
    `excluded_missing_year` holds the connected links dropped for lacking one.
    """
    links = links.copy()
    links["station_id"] = links["station_id"].astype(str).str.strip()
    links["dam_id"] = links["dam_id"].astype(str).str.strip()

    connected = links[links["relationship"].isin(CONNECTED_RELATIONSHIPS)].copy()

    removal_year = pd.to_numeric(connected["year_removed"], errors="coerce")
    has_year = removal_year.notna() & (removal_year != 0)

    excluded_missing_year = (
        connected.loc[~has_year, ["dam_id", "dam_name", "station_id"]]
        .drop_duplicates()
    )

    eligible_links = connected.loc[has_year].copy()
    eligible_links["removal_year"] = removal_year.loc[has_year].astype(int)

    pairs = (
        eligible_links[["dam_id", "dam_name", "station_id", "removal_year"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return pairs, excluded_missing_year


def prepare_results(results, characteristics):
    """Filter raw WQP results down to the configured characteristics and
    valid, dated observations."""
    results = results.copy()
    results["station_id"] = results["MonitoringLocationIdentifier"].astype(str).str.strip()
    results["ActivityStartDate"] = pd.to_datetime(results["ActivityStartDate"], errors="coerce")
    results = results.dropna(subset=["ActivityStartDate"])
    results["measurement_year"] = results["ActivityStartDate"].dt.year
    results = results[results["CharacteristicName"].isin(characteristics)].copy()
    return results


def classify_period(measurement_year, removal_year, years_before, years_after):
    """Classify a single measurement year as 'pre', 'post', or neither (None).

    The removal year itself (offset == 0) always returns None -- see the
    module docstring for why it's excluded from both windows.
    """
    offset = measurement_year - removal_year
    if -years_before <= offset <= -1:
        return "pre"
    if 1 <= offset <= years_after:
        return "post"
    return None


def build_observation_counts(pairs, results, years_before, years_after):
    """Long-format table: one row per (dam_id, station_id, CharacteristicName,
    period) that has at least one qualifying observation."""
    columns = ["dam_id", "dam_name", "station_id", "CharacteristicName", "period", "n_observations"]

    if pairs.empty or results.empty:
        return pd.DataFrame(columns=columns)

    merged = pairs.merge(
        results[["station_id", "CharacteristicName", "measurement_year"]],
        on="station_id",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["period"] = [
        classify_period(year, removal, years_before, years_after)
        for year, removal in zip(merged["measurement_year"], merged["removal_year"])
    ]
    merged = merged.dropna(subset=["period"])

    if merged.empty:
        return pd.DataFrame(columns=columns)

    counts = (
        merged.groupby(["dam_id", "dam_name", "station_id", "CharacteristicName", "period"])
        .size()
        .reset_index(name="n_observations")
    )
    return counts


def build_pair_report(pairs, counts):
    """One row per (dam_id, station_id) pair with pre/post/both flags.

    'has_both' requires observations on that SAME pair -- this is the
    pair-level unit everything else rolls up from.
    """
    if counts.empty:
        pivot = pd.DataFrame(columns=["dam_id", "station_id", "pre", "post"])
    else:
        pivot = (
            counts.groupby(["dam_id", "station_id", "period"])["n_observations"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ("pre", "post"):
            if col not in pivot.columns:
                pivot[col] = 0

    report = pairs.merge(pivot, on=["dam_id", "station_id"], how="left")
    report["pre"] = report["pre"].fillna(0).astype(int)
    report["post"] = report["post"].fillna(0).astype(int)
    report["has_pre"] = report["pre"] > 0
    report["has_post"] = report["post"] > 0
    report["has_both"] = report["has_pre"] & report["has_post"]
    report = report.rename(
        columns={"pre": "n_pre_observations", "post": "n_post_observations"}
    )
    return report


def build_dam_report(pair_report):
    """One row per dam.

    'has_both' is "any single linked pair has both" -- NOT "any pair has
    pre" AND "any (possibly different) pair has post". See module docstring.
    """
    if pair_report.empty:
        return pd.DataFrame(columns=[
            "dam_id", "dam_name", "n_linked_stations", "has_pre", "has_post",
            "has_both", "pairs_with_both", "n_pre_observations", "n_post_observations",
        ])

    dam_report = (
        pair_report.groupby(["dam_id", "dam_name"])
        .agg(
            n_linked_stations=("station_id", "nunique"),
            has_pre=("has_pre", "any"),
            has_post=("has_post", "any"),
            has_both=("has_both", "any"),
            pairs_with_both=("has_both", "sum"),
            n_pre_observations=("n_pre_observations", "sum"),
            n_post_observations=("n_post_observations", "sum"),
        )
        .reset_index()
    )
    return dam_report


def build_characteristic_report(counts):
    """Same pre/post/both logic, computed separately per CharacteristicName
    so results are never pooled across different measured quantities."""
    columns = [
        "CharacteristicName", "pairs_with_pre", "pairs_with_post", "pairs_with_both",
        "dams_with_pre", "dams_with_post", "dams_with_both",
        "n_pre_observations", "n_post_observations",
    ]

    if counts.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for characteristic, group in counts.groupby("CharacteristicName"):
        pivot = (
            group.groupby(["dam_id", "station_id", "period"])["n_observations"]
            .sum()
            .unstack(fill_value=0)
        )
        for col in ("pre", "post"):
            if col not in pivot.columns:
                pivot[col] = 0

        pair_flags = pivot.reset_index()
        pair_flags["has_pre"] = pair_flags["pre"] > 0
        pair_flags["has_post"] = pair_flags["post"] > 0
        pair_flags["has_both"] = pair_flags["has_pre"] & pair_flags["has_post"]

        dam_flags = pair_flags.groupby("dam_id")[["has_pre", "has_post", "has_both"]].any()

        rows.append({
            "CharacteristicName": characteristic,
            "pairs_with_pre": int(pair_flags["has_pre"].sum()),
            "pairs_with_post": int(pair_flags["has_post"].sum()),
            "pairs_with_both": int(pair_flags["has_both"].sum()),
            "dams_with_pre": int(dam_flags["has_pre"].sum()),
            "dams_with_post": int(dam_flags["has_post"].sum()),
            "dams_with_both": int(dam_flags["has_both"].sum()),
            "n_pre_observations": int(group.loc[group["period"] == "pre", "n_observations"].sum()),
            "n_post_observations": int(group.loc[group["period"] == "post", "n_observations"].sum()),
        })

    return pd.DataFrame(rows, columns=columns).sort_values("CharacteristicName").reset_index(drop=True)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_links(path):
    df = pd.read_csv(path, dtype={"station_id": "string", "dam_id": "string"})
    require_columns(df, LINKS_REQUIRED_COLUMNS, f"Dam-station link table ({path})")
    return df


def load_results(results_dir):
    batch_files = sorted(Path(results_dir).glob("results_batch_*.csv"))
    frames = [
        pd.read_csv(f, low_memory=False, dtype={"MonitoringLocationIdentifier": "string"})
        for f in batch_files
    ]
    results = pd.concat(frames, ignore_index=True)
    require_columns(results, RESULTS_REQUIRED_COLUMNS, "WQP results")
    return results


def _write_manifest(manifest_path, row):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    full_row = {field: row.get(field, "") for field in MANIFEST_FIELDS}
    pd.DataFrame([full_row]).to_csv(manifest_path, index=False)


def process_huc8(cfg, huc8, results):
    links_path = cfg.project_root / "data" / "processed" / f"dam_station_links_{huc8}.csv"

    output_dir = cfg.project_root / "data" / "processed"
    pairs_path = output_dir / f"pilot_eligibility_pairs_{huc8}.csv"
    dams_path = output_dir / f"pilot_eligibility_dams_{huc8}.csv"
    characteristics_path = output_dir / f"pilot_eligibility_by_characteristic_{huc8}.csv"
    manifest_path = output_dir / f"pilot_eligibility_manifest_{huc8}.csv"

    wq_window = cfg.get("wq_window", {}) or {}
    years_before = int(wq_window.get("years_before", 3))
    years_after = int(wq_window.get("years_after", 3))
    characteristics = cfg.get("wq_characteristics", [])
    removal_year_handling = "excluded from both pre- and post- windows"

    base_manifest = {
        "huc8": huc8,
        "years_before": years_before,
        "years_after": years_after,
        "removal_year_handling": removal_year_handling,
        "characteristics": ";".join(characteristics),
    }

    try:
        if not links_path.exists():
            raise FileNotFoundError(
                f"Missing input: dam-station link table not found at {links_path}. "
                "Run link_dams_stations.py for this HUC8 first."
            )

        links = load_links(links_path)

        pairs_universe, excluded_missing_year = prepare_links(links)
        filtered_results = prepare_results(results, characteristics)

        counts = build_observation_counts(pairs_universe, filtered_results, years_before, years_after)
        pair_report = build_pair_report(pairs_universe, counts)
        dam_report = build_dam_report(pair_report)
        characteristic_report = build_characteristic_report(counts)

        output_dir.mkdir(parents=True, exist_ok=True)
        pair_report.to_csv(pairs_path, index=False)
        dam_report.to_csv(dams_path, index=False)
        characteristic_report.to_csv(characteristics_path, index=False)

        manifest_row = {
            **base_manifest,
            "status": "ok",
            "dams_total_linked": int(links["dam_id"].astype(str).nunique()),
            "dams_excluded_missing_removal_year": int(excluded_missing_year["dam_id"].nunique()),
            "pairs_considered": int(len(pairs_universe)),
            "dams_eligible_pre": int(dam_report["has_pre"].sum()),
            "dams_eligible_post": int(dam_report["has_post"].sum()),
            "dams_eligible_both": int(dam_report["has_both"].sum()),
            "pairs_eligible_pre": int(pair_report["has_pre"].sum()),
            "pairs_eligible_post": int(pair_report["has_post"].sum()),
            "pairs_eligible_both": int(pair_report["has_both"].sum()),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "error_message": "",
        }
        _write_manifest(manifest_path, manifest_row)

        print(f"HUC8: {huc8}")
        print(
            f"Dams eligible (pre/post/both): "
            f"{manifest_row['dams_eligible_pre']}/{manifest_row['dams_eligible_post']}/"
            f"{manifest_row['dams_eligible_both']} of {manifest_row['dams_total_linked']} linked dams "
            f"({manifest_row['dams_excluded_missing_removal_year']} excluded for missing removal year)"
        )
        print(
            f"Pairs eligible (pre/post/both): "
            f"{manifest_row['pairs_eligible_pre']}/{manifest_row['pairs_eligible_post']}/"
            f"{manifest_row['pairs_eligible_both']} of {manifest_row['pairs_considered']} considered pairs"
        )
        print(f"Pair-level report: {pairs_path}")
        print(f"Dam-level report: {dams_path}")
        print(f"By-characteristic report: {characteristics_path}")
        print(f"Manifest: {manifest_path}")

    except Exception as exc:
        status = "missing_input" if isinstance(exc, FileNotFoundError) else "failed"
        _write_manifest(manifest_path, {
            **base_manifest,
            "status": status,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "error_message": str(exc),
        })
        print(f"FAILED ({status}): {exc}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    parser.add_argument(
        "--huc8",
        default=None,
        help="Process only this HUC8. Defaults to every HUC8 listed in the "
        "case-study config, processed one at a time.",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    huc8_list = [args.huc8] if args.huc8 else cfg.huc8s

    # The WQP results directory is shared across the whole case study
    # (download_wqp_results.py downloads for every candidate station at
    # once), not per-HUC8 -- so it's checked and loaded once here, unlike
    # the per-HUC8 dam-station link table.
    results_dir = cfg.project_root / "data" / "raw" / "wqp" / "results" / "pilot_download"

    if not results_dir.exists() or not any(results_dir.glob("results_batch_*.csv")):
        error = FileNotFoundError(
            f"Missing input: no WQP result batches found under {results_dir}. "
            "Run download_wqp_results.py first."
        )
        output_dir = cfg.project_root / "data" / "processed"
        for huc8 in huc8_list:
            _write_manifest(output_dir / f"pilot_eligibility_manifest_{huc8}.csv", {
                "huc8": huc8,
                "status": "missing_input",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "error_message": str(error),
            })
        print(f"FAILED (missing_input): {error}", file=sys.stderr)
        raise error

    results = load_results(results_dir)

    outcomes = []

    for huc8 in huc8_list:
        try:
            process_huc8(cfg, huc8, results)
            outcomes.append((huc8, "ok"))
        except Exception as exc:
            outcomes.append((huc8, f"failed: {exc}"))

    if len(huc8_list) > 1:
        print("\n=== Summary ===")
        for huc8, status in outcomes:
            print(f"  {huc8}: {status}")

    if any(status != "ok" for _, status in outcomes):
        raise RuntimeError("One or more HUC8s failed; see log above.")


if __name__ == "__main__":
    main()
