"""
Select WQP stations in scope for this case study's watershed(s).

This is the sole, canonical generator of
`data/raw/wqp/results/<case_study>/pilot_station_selection.csv`. Nothing else should
write to that path.

An earlier version of this script (and, before that, two conflicting
notebook cells) narrowed this list by straight-line distance to the
nearest dam. That logic has been removed entirely: distance "as the crow
flies" is not a reliable proxy for hydrological connectivity -- a station
can be close to a dam in a straight line but in an unrelated watershed, or
far in a straight line but closely connected along a winding river. The
only correct determination of relevance is network-based:
`snap_features.py` snaps every in-scope station (and dam) onto the
NHDPlus flowline network, and `link_dams_stations.py` walks the directed
flow graph to classify each station as upstream, downstream, same-reach,
or unconnected relative to a dam, with a network-distance estimate. This
script's only job is to assemble the full set of stations that network
step gets to evaluate -- every surface-water station downloaded for this
case study's HUC8(s) -- with no additional geographic filtering. There is
no "pass 1 narrows, pass 2 confirms" split anymore: this is just scoping,
and the network step is the sole arbiter of relevance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common.config import default_case_study, load_config

DEFAULT_STATION_TYPES = ["Stream", "River/Stream"]

OUTPUT_COLUMNS_REQUIRED = {"MonitoringLocationIdentifier", "source_huc8"}


def load_scope_stations(station_dir, huc8s, station_types):
    """Load raw downloaded WQP station files for this case study's HUC8s.

    Each file is the per-HUC8 output of download_wqp_stations.py, named
    "<huc8>.csv". Loading by exact filename (rather than globbing every
    file in the directory) means a case study's station set is only ever
    built from the HUC8s it actually declares.
    """
    station_dir = Path(station_dir)
    frames = []
    missing = []

    for huc8 in huc8s:
        path = station_dir / f"{huc8}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path, low_memory=False)
        df["source_huc8"] = huc8
        frames.append(df)

    if missing:
        raise FileNotFoundError(
            "Missing downloaded WQP station file(s) for this case study's "
            f"HUC8 scope: {missing}. Run download_wqp_stations.py first."
        )

    stations = pd.concat(frames, ignore_index=True)

    stations = stations[
        stations["MonitoringLocationTypeName"].isin(station_types)
    ].copy()

    for col in ["LatitudeMeasure", "LongitudeMeasure"]:
        stations[col] = pd.to_numeric(stations[col], errors="coerce")

    stations = stations.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])
    stations = stations[
        stations["LatitudeMeasure"].between(-90, 90)
        & stations["LongitudeMeasure"].between(-180, 180)
    ].copy()
    stations = stations.drop_duplicates(subset="MonitoringLocationIdentifier")

    return stations


def validate_stations(stations):
    """Validate a completed station table before it is allowed to be written.

    This is the guard against overwriting the production file with a
    broken or empty result -- failures here must not touch disk.
    """
    missing_cols = OUTPUT_COLUMNS_REQUIRED - set(stations.columns)
    if missing_cols:
        raise ValueError(
            f"Station table is missing expected columns: {sorted(missing_cols)}"
        )

    if stations.empty:
        raise ValueError(
            "No in-scope surface-water stations found. Refusing to "
            "overwrite the existing station file with an empty result."
        )

    if stations["MonitoringLocationIdentifier"].isna().any():
        raise ValueError(
            "Station table has rows with a missing MonitoringLocationIdentifier."
        )


def _write_failure_manifest(manifest_path, cfg_name, error):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "status": "failed",
        "case_study": cfg_name,
        "huc8s": "",
        "station_count": "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "error_message": str(error),
    }]).to_csv(manifest_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)

    results_dir = cfg.wqp_results_dir
    output_path = results_dir / "pilot_station_selection.csv"
    manifest_path = results_dir / "pilot_station_selection_manifest.csv"

    try:
        print(f"Case study: {cfg.name}")
        print(f"HUC8 scope: {cfg.huc8s}")

        station_dir = cfg.project_root / "data" / "raw" / "wqp" / "stations"
        station_types = cfg.get("station_types", DEFAULT_STATION_TYPES)
        stations = load_scope_stations(station_dir, cfg.huc8s, station_types)

        validate_stations(stations)

        # Write to a temp file first; only replace the production output
        # once the result above has validated successfully.
        results_dir.mkdir(parents=True, exist_ok=True)

        tmp_output = output_path.with_suffix(".tmp.csv")
        stations.to_csv(tmp_output, index=False)
        tmp_output.replace(output_path)

        manifest_row = {
            "status": "ok",
            "case_study": cfg.name,
            "huc8s": ",".join(cfg.huc8s),
            "station_count": int(stations["MonitoringLocationIdentifier"].nunique()),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "error_message": "",
        }
        pd.DataFrame([manifest_row]).to_csv(manifest_path, index=False)

        print(
            f"\nSelected {manifest_row['station_count']:,} in-scope surface-water "
            "stations (no geographic filtering -- network connectivity in "
            "snap_features.py / link_dams_stations.py determines relevance)."
        )
        print(f"Saved: {output_path}")
        print(f"Manifest: {manifest_path}")

    except Exception as exc:
        _write_failure_manifest(manifest_path, args.case_study, exc)
        print(f"FAILED: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
