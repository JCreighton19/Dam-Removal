"""
Compare a validation snapping run with the current snapped features,
before the new outputs replace them.

    python etl/snap_features.py --output-dir data/processed/snap_validation \
        --huc8 <HUC8s with known failures> --sample 30
    python etl/compare_snapping.py --audit-dir \
        data/processed/preprocessing_audit/all_removals

For every dam and station in the new run's HUC8s, one row with the old
reach (data/processed/snapped_features_<huc8>.gpkg, *_snapped layers; a
point snapped to several reaches keeps the nearest, as load_core_tables
does) and the new decision (snap_diagnostics_<huc8>.csv):

    change   same_reach | moved_reach | now_ambiguous | now_rejected |
             newly_accepted | ambiguous_not_in_old | rejected_not_in_old

Known failures, with what the new run decided for each:
- KNOWN_WRONG_STATIONS: confirmed on the wrong reach on the audit map
  (notebooks/preprocessing_audit.ipynb, section 14);
- with --audit-dir: the audit's dam_snap_check.csv and
  station_snap_check.csv verdicts ('likely wrong', 'check').

Writes snap_comparison.csv to the new run's folder and prints summaries.
Reads only; nothing in data/processed/ is changed.

--print-known-huc8s prints the HUC8s holding the known and audit-flagged
failures (from the current warehouse), for snap_features --huc8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from etl.common.config import default_case_study, load_config
from etl.common.warehouse import warehouse_dir

KNOWN_WRONG_STATIONS = [
    "11113300-SILHARSB1", "11113300-SILHARL1", "11113300-SILHARL2",
    "CHESAPEAKEMONITORINGCOOP-CMC.LWQVC.MILL3", "CADWR-1541.00", "MNPCA-S004-561",
    "WIDNR_WQX-10040927",
]

LAYERS = {"dam": ("dams_snapped", "AR_ID"), "station": ("stations_snapped",
                                                        "MonitoringLocationIdentifier")}


def old_assignments(path):
    """One row per (feature_kind, feature_id) from a current snapped file."""
    if not path.exists():
        return pd.DataFrame(columns=["feature_kind", "feature_id", "old_reach", "old_distance_m"])
    frames = []
    for kind, (layer, id_column) in LAYERS.items():
        frame = gpd.read_file(path, layer=layer, ignore_geometry=True,
                              columns=[id_column, "NHDPlusID", "snap_distance_m"])
        frame = frame.sort_values([id_column, "snap_distance_m", "NHDPlusID"]) \
            .drop_duplicates(id_column)
        frames.append(pd.DataFrame({
            "feature_kind": kind,
            "feature_id": frame[id_column].astype(str),
            "old_reach": pd.to_numeric(frame["NHDPlusID"], errors="coerce").astype("Int64"),
            "old_distance_m": frame["snap_distance_m"],
        }))
    return pd.concat(frames, ignore_index=True)


def classify(row):
    in_old = not pd.isna(row["old_reach"])
    if row["snap_status"] != "accepted":
        return f"now_{row['snap_status']}" if in_old else f"{row['snap_status']}_not_in_old"
    if not in_old:
        return "newly_accepted"
    return "same_reach" if row["NHDPlusID"] == row["old_reach"] else "moved_reach"


def compare(old_dir, new_dir, huc8s):
    rows = []
    for huc8 in huc8s:
        new = pd.read_csv(new_dir / f"snap_diagnostics_{huc8}.csv",
                          dtype={"feature_id": str, "huc8": str})
        new["NHDPlusID"] = pd.to_numeric(new["NHDPlusID"], errors="coerce").astype("Int64")
        old = old_assignments(old_dir / f"snapped_features_{huc8}.gpkg")
        merged = new.merge(old, on=["feature_kind", "feature_id"], how="left")
        merged["change"] = merged.apply(classify, axis=1)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def known_failures(comparison, audit_dir):
    """Rows for every known or audit-flagged failure present in the run."""
    known = [pd.DataFrame({"feature_kind": "station", "feature_id": KNOWN_WRONG_STATIONS,
                           "audit_verdict": "confirmed wrong (map)"})]
    if audit_dir:
        for kind, name, id_column in [("dam", "dam_snap_check.csv", "AR_ID"),
                                      ("station", "station_snap_check.csv", "station_id")]:
            path = Path(audit_dir) / name
            if not path.exists():
                print(f"(no {path}; skipped)", file=sys.stderr)
                continue
            audit = pd.read_csv(path, dtype={id_column: str})
            audit = audit[audit["verdict"].isin(["likely wrong", "check", "confirmed wrong (map)"])]
            known.append(pd.DataFrame({"feature_kind": kind, "feature_id": audit[id_column],
                                       "audit_verdict": audit["verdict"]}))
    known = pd.concat(known, ignore_index=True).drop_duplicates(["feature_kind", "feature_id"])
    found = known.merge(comparison, on=["feature_kind", "feature_id"], how="inner")
    missing = len(known) - len(found)
    return found, missing


def known_failure_huc8s(cfg, audit_dir):
    """HUC8s of the known / audit-flagged dams and stations, looked up in
    the current warehouse tables."""
    out = warehouse_dir(cfg)
    stations = pd.read_parquet(out / "wq_station.parquet", columns=["station_id", "huc8"])
    dams = pd.read_parquet(out / "dam_removal.parquet", columns=["AR_ID", "HUC8"])
    found, _ = known_failures(
        pd.concat([
            pd.DataFrame({"feature_kind": "station", "feature_id": stations["station_id"],
                          "huc8": stations["huc8"]}),
            pd.DataFrame({"feature_kind": "dam", "feature_id": dams["AR_ID"], "huc8": dams["HUC8"]}),
        ], ignore_index=True),
        audit_dir,
    )
    return sorted(found["huc8"].dropna().unique())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--old-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--new-dir", type=Path, default=Path("data/processed/snap_validation"))
    parser.add_argument("--audit-dir", type=Path, default=None,
                        help="Folder with the preprocessing audit's dam_snap_check.csv and "
                        "station_snap_check.csv.")
    parser.add_argument("--print-known-huc8s", action="store_true",
                        help="Only print the HUC8s holding known / audit-flagged failures.")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    resolve = lambda p: p if p.is_absolute() else cfg.project_root / p  # noqa: E731
    old_dir, new_dir = resolve(args.old_dir), resolve(args.new_dir)
    audit_dir = resolve(args.audit_dir) if args.audit_dir else None
    if args.print_known_huc8s:
        print(" ".join(known_failure_huc8s(cfg, audit_dir)))
        return
    huc8s = sorted(p.stem.removeprefix("snap_diagnostics_")
                   for p in new_dir.glob("snap_diagnostics_*.csv"))
    if not huc8s:
        raise FileNotFoundError(f"No snap_diagnostics_*.csv in {new_dir} (run snap_features "
                                "with --output-dir first)")

    comparison = compare(old_dir, new_dir, huc8s)
    path = new_dir / "snap_comparison.csv"
    comparison.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    print(f"HUC8s compared: {len(huc8s)}")
    print("\nOld vs new, by kind:")
    print(pd.crosstab(comparison["change"], comparison["feature_kind"], margins=True).to_string())
    print("\nNew decisions, by kind and reason:")
    print(comparison.groupby(["feature_kind", "snap_status", "snap_reason"]).size()
          .rename("points").to_string())
    print("\nDatums (stations):")
    stations = comparison[comparison["feature_kind"] == "station"]
    print(stations.groupby(["datum_source", "datum_assumed"], dropna=False).size()
          .rename("stations").to_string())
    moved = comparison[comparison["change"] == "moved_reach"]
    if len(moved):
        print("\nMoved to another reach (first 25):")
        print(moved[["feature_kind", "feature_id", "feature_name", "old_reach", "old_distance_m",
                     "NHDPlusID", "snap_distance_m", "reach_gnis_name", "snap_reason"]]
              .head(25).to_string(index=False))

    found, missing = known_failures(comparison, audit_dir)
    print(f"\nKnown / audit-flagged failures in these HUC8s: {len(found)} "
          f"({missing} not in the HUC8s run)")
    if len(found):
        print(pd.crosstab([found["feature_kind"], found["audit_verdict"]], found["change"])
              .to_string())
        print(found[["feature_kind", "feature_id", "audit_verdict", "change", "snap_status",
                     "snap_reason", "old_reach", "NHDPlusID", "reach_gnis_name"]]
              .to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
