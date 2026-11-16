"""
Turn the verdicts of one or more reviewed station GeoPackages
(analysis/audits/export_station_review.py, edited in QGIS) into rows of
config/snap_overrides.csv, and save every verdict for the error-rate
estimate. Runs locally. Workflow: docs/station_review.md.

The stations layer's review fields:

    verdict             correct | wrong_reach | wrong_side | not_a_stream | unsure
    correct_nhdplus_id  wrong_reach only: the right flowline (in the
                        GeoPackage's flowlines layer, within
                        snapping.OVERRIDE_MAX_M of the station)
    review_note         free text; required for wrong_side

    wrong_reach  -> station,<id>,accept,<correct_nhdplus_id>,station review: wrong reach; <note>,...
    not_a_stream -> station,<id>,reject,,station review: not a stream station; <note>,...
    wrong_side   -> listed, not written: the fix depends on the cause (the
                    station, the dam's placement or the reservoir), decided
                    case by case
    correct, unsure -> no override

Several GeoPackages (e.g. the BACI pass in one file and the rest in a later
export) are merged by station_id; a station with different verdicts in two
files is an error. Checks are all reported at once and nothing is written if
any fails. Existing station override rows for the same stations are replaced
(and listed); every other row and the file's comment lines are kept. The new
file is validated with snapping.load_overrides before it replaces the old one.

Also writes --verdicts (default: station_review_verdicts.csv next to the
first GeoPackage): one row per reviewed station with its tier, sample flag
and weight, for analysis/audits/station_review_errors.py.

    python etl/apply_station_review.py --review a.gpkg b.gpkg --reviewer jc --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from etl.apply_dam_review import COLUMNS, write_overrides
from etl.common import snapping as sn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = PROJECT_ROOT / "config" / "snap_overrides.csv"
METRIC_CRS = "EPSG:5070"
VERDICTS = ("correct", "wrong_reach", "wrong_side", "not_a_stream", "unsure")
KEEP = ["station_id", "verdict", "correct_nhdplus_id", "review_note", "inspect_reason",
        "risk_tier", "role", "baci_station", "in_sample", "sample_weight", "stratum", "dam_ids",
        "NHDPlusID", "snap_reason", "MonitoringLocationName"]


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def read_reviews(paths):
    """(reviewed stations with geometry and source file, flowline ids and
    geometries per file). Only rows with a verdict are kept."""
    frames, lines = [], {}
    for path in paths:
        stations = gpd.read_file(path, layer="stations")
        stations["verdict"] = stations["verdict"].map(_text).str.lower()
        stations = stations[stations["verdict"].ne("")].copy()
        stations["source_file"] = Path(path).name
        frames.append(stations)
        flow = gpd.read_file(path, layer="flowlines", columns=["nhdplus_id"])
        lines[Path(path).name] = flow
    if not frames:
        return gpd.GeoDataFrame(), lines
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs), lines


def check_reviews(reviews, lines):
    """Every problem, as messages."""
    errors = []
    by_station = reviews.groupby("station_id")["verdict"].nunique()
    for station in by_station[by_station > 1].index:
        files = reviews.loc[reviews["station_id"].eq(station), ["source_file", "verdict"]]
        errors.append(f"{station}: different verdicts in different files "
                      f"{files.values.tolist()}")
    metric = reviews.to_crs(METRIC_CRS)
    for index, row in reviews.iterrows():
        where = f"{row['station_id']} ({row['source_file']})"
        verdict = row["verdict"]
        if verdict not in VERDICTS:
            errors.append(f"{where}: verdict '{verdict}' is not one of {', '.join(VERDICTS)}")
            continue
        reach = pd.to_numeric(row.get("correct_nhdplus_id"), errors="coerce")
        if verdict == "wrong_reach":
            if pd.isna(reach):
                errors.append(f"{where}: wrong_reach needs correct_nhdplus_id")
                continue
            if not pd.isna(pd.to_numeric(row.get("NHDPlusID"), errors="coerce")) \
                    and int(reach) == int(pd.to_numeric(row["NHDPlusID"])):
                errors.append(f"{where}: correct_nhdplus_id {int(reach)} is the current reach")
                continue
            flow = lines[row["source_file"]]
            match = flow[pd.to_numeric(flow["nhdplus_id"], errors="coerce").eq(reach)]
            if match.empty:
                errors.append(f"{where}: {int(reach)} is not in the flowlines layer")
                continue
            distance = float(match.to_crs(METRIC_CRS).distance(metric.geometry.loc[index]).min())
            if distance > sn.OVERRIDE_MAX_M:
                errors.append(f"{where}: flowline {int(reach)} is {distance:,.0f} m from the "
                              f"station (at most {sn.OVERRIDE_MAX_M:,} m)")
        elif not pd.isna(reach):
            errors.append(f"{where}: correct_nhdplus_id is only for wrong_reach")
        if verdict == "wrong_side" and not _text(row.get("review_note")):
            errors.append(f"{where}: wrong_side needs a review_note (which dam, and why)")
    return errors


def override_rows(reviews, reviewer, reviewed_on):
    """snap_overrides rows for wrong_reach (accept) and not_a_stream (reject)."""
    rows = []
    for _, row in reviews.drop_duplicates("station_id").iterrows():
        note = _text(row.get("review_note")).replace("\n", " ")
        if row["verdict"] == "wrong_reach":
            action, reach, what = "accept", str(int(pd.to_numeric(row["correct_nhdplus_id"]))), \
                "wrong reach"
        elif row["verdict"] == "not_a_stream":
            action, reach, what = "reject", "", "not a stream station"
        else:
            continue
        rows.append({"feature_kind": "station", "feature_id": str(row["station_id"]),
                     "action": action, "nhdplus_id": reach,
                     "reason": f"station review: {what}" + (f"; {note}" if note else ""),
                     "reviewed_by": reviewer, "reviewed_on": reviewed_on,
                     "corrected_lat": "", "corrected_lon": ""})
    return pd.DataFrame(rows, columns=COLUMNS)


def merge_station_overrides(path, new):
    """(comment lines, merged rows, replaced station ids)."""
    path = Path(path)
    comments, existing = [], pd.DataFrame(columns=COLUMNS)
    if path.exists():
        comments = [line for line in path.read_text().splitlines() if line.startswith("#")]
        existing = pd.read_csv(path, dtype=str, comment="#", keep_default_na=False)
    for column in COLUMNS:
        if column not in existing:
            existing[column] = ""
    same = existing["feature_kind"].eq("station") & existing["feature_id"].isin(set(new["feature_id"]))
    replaced = sorted(existing.loc[same, "feature_id"])
    merged = pd.concat([existing.loc[~same, COLUMNS], new[COLUMNS]], ignore_index=True)
    return comments, merged, replaced


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", type=Path, nargs="+", required=True,
                        help="The edited review GeoPackage(s).")
    parser.add_argument("--reviewer", required=True, help="Initials for reviewed_by.")
    parser.add_argument("--reviewed-on", default=date.today().isoformat())
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Check and show; write nothing.")
    args = parser.parse_args()

    reviews, lines = read_reviews(args.review)
    if reviews.empty:
        print("No station has a verdict yet: nothing to do.")
        return
    errors = check_reviews(reviews, lines)
    if errors:
        print(f"{len(errors)} problem(s); nothing written:\n  - " + "\n  - ".join(errors))
        sys.exit(1)

    unique = reviews.drop_duplicates("station_id")
    print(f"Reviewed stations: {len(unique)}")
    print(pd.crosstab(unique["inspect_reason"].fillna("?"), unique["verdict"], margins=True).to_string())
    new = override_rows(unique, args.reviewer, args.reviewed_on)
    comments, merged, replaced = merge_station_overrides(args.overrides, new)
    print(f"\nOverride rows: {len(new)} "
          f"({new['action'].value_counts().to_dict() if len(new) else {}})")
    if replaced:
        print(f"Replacing earlier station overrides for: {replaced}")
    side = unique[unique["verdict"].eq("wrong_side")]
    if len(side):
        print(f"\nwrong_side, not written (decide case by case): {len(side)}")
        print(side[["station_id", "dam_ids", "role", "review_note"]].to_string(index=False))
    if len(new):
        print("\n" + new[["feature_id", "action", "nhdplus_id", "reason"]].to_string(index=False))

    verdicts = args.verdicts or args.review[0].with_name("station_review_verdicts.csv")
    if args.dry_run:
        print("\nDry run: nothing written.")
        return
    pd.DataFrame(unique[[c for c in KEEP + ["source_file"] if c in unique]]).to_csv(verdicts, index=False)
    print(f"\nWrote {verdicts}")
    if len(new):
        write_overrides(args.overrides, comments, merged)
        print(f"Wrote {args.overrides} ({len(merged)} override rows).")


if __name__ == "__main__":
    main()
