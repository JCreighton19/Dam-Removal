"""
Turn the corrections layer of a reviewed dam GeoPackage
(analysis/audits/export_dam_review.py, edited in QGIS) into rows of
config/snap_overrides.csv. Runs locally. Workflow: docs/dam_review.md.

Each point in the corrections layer is one reviewed dam:

    AR_ID        the dam (must be in the GeoPackage's dams layer, once)
    action       move   -> the dam is at this point, on flowline nhdplus_id
                 reject -> no reach can be identified; keep it out
    nhdplus_id   move only: the flowline the point was snapped to
    reason       required: what the review found
    source       optional: imagery, Street View, a news article, ...

    move   -> dam,<AR_ID>,accept,<nhdplus_id>,<reason [source]>,<reviewer>,<date>,<lat>,<lon>
    reject -> dam,<AR_ID>,reject,,<reason [source]>,<reviewer>,<date>,,

Checks, all reported at once; nothing is written if any fails: the fields
above; a move point lies within ON_REACH_M of its flowline in the
GeoPackage's flowlines layer. Existing override rows for the same dams are
replaced (and listed); every other row and the file's comment lines are
kept. The new file is validated with snapping.load_overrides before it
replaces the old one, and match_dams.py checks the reaches again on the
cluster.

    python etl/apply_dam_review.py --review ~/Desktop/dam_review.gpkg --reviewer jc
    python etl/apply_dam_review.py --review ... --reviewer jc --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from etl.common import snapping as sn
from etl.common.reach_inputs import dam_id_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = PROJECT_ROOT / "config" / "snap_overrides.csv"
METRIC_CRS = "EPSG:5070"
# Stricter than match_dams.py (snapping.OVERRIDE_ON_REACH_M): a point snapped
# to the line in QGIS is on it.
ON_REACH_M = 25
ACTIONS = {"move": "accept", "reject": "reject"}
COLUMNS = sn.OVERRIDE_COLUMNS + sn.OVERRIDE_COORDINATE_COLUMNS


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def read_review(path):
    """(corrections, dam ids, flowlines) from a review GeoPackage."""
    corrections = gpd.read_file(path, layer="corrections")
    dams = gpd.read_file(path, layer="dams", columns=["AR_ID"], ignore_geometry=True)
    flowlines = gpd.read_file(path, layer="flowlines", columns=["nhdplus_id"])
    return corrections, set(dam_id_text(dams["AR_ID"])), flowlines


def check_corrections(corrections, dam_ids, flowlines):
    """List of problems (empty = valid)."""
    errors = []
    lines = flowlines.assign(nhdplus_id=pd.to_numeric(flowlines["nhdplus_id"], errors="coerce")) \
        .dropna(subset=["nhdplus_id"]).drop_duplicates("nhdplus_id")
    lines = lines.set_index(lines["nhdplus_id"].astype("int64")).to_crs(METRIC_CRS).geometry
    points = corrections.to_crs(METRIC_CRS).geometry if len(corrections) else corrections.geometry
    ids = dam_id_text(corrections["AR_ID"].astype("string").fillna("")) if len(corrections) else []
    for number, (index, row) in enumerate(corrections.iterrows(), start=1):
        dam = ids.loc[index] if len(corrections) else ""
        where = f"correction {number} ({dam or 'no AR_ID'})"
        action = _text(row.get("action")).lower()
        reach = pd.to_numeric(pd.Series([row.get("nhdplus_id")]), errors="coerce").iloc[0]
        if not dam:
            errors.append(f"{where}: AR_ID is empty")
        elif dam not in dam_ids:
            errors.append(f"{where}: AR_ID not in the review's dams layer")
        if action not in ACTIONS:
            errors.append(f"{where}: action {action!r} must be one of {sorted(ACTIONS)}")
        if not _text(row.get("reason")):
            errors.append(f"{where}: reason is empty")
        if action == "reject" and pd.notna(reach):
            errors.append(f"{where}: reject must not give an nhdplus_id")
        if action == "move":
            point = points.loc[index]
            if point is None or point.is_empty or point.geom_type != "Point":
                errors.append(f"{where}: move needs a point")
            elif pd.isna(reach) or reach != int(reach):
                errors.append(f"{where}: move needs the integer nhdplus_id of the flowline")
            elif int(reach) not in lines.index:
                errors.append(f"{where}: nhdplus_id {int(reach)} is not in the flowlines layer")
            else:
                distance = point.distance(lines.loc[int(reach)])
                if distance > ON_REACH_M:
                    errors.append(f"{where}: point is {distance:,.0f} m from flowline {int(reach)} "
                                  f"(at most {ON_REACH_M} m; snap it to the line)")
    duplicated = pd.Series(list(ids)).duplicated(keep=False) if len(corrections) else pd.Series([], dtype=bool)
    if duplicated.any():
        errors.append(f"more than one correction for {sorted(set(pd.Series(list(ids))[duplicated]))}")
    return errors


def override_rows(corrections, reviewer, reviewed_on):
    """New snap_overrides rows (strings), one per correction."""
    wgs = corrections.to_crs("EPSG:4326")
    rows = []
    for index, row in corrections.iterrows():
        action = _text(row["action"]).lower()
        reason = _text(row["reason"])
        source = _text(row.get("source"))
        if source:
            reason = f"{reason} [source: {source}]"
        moved = action == "move"
        point = wgs.geometry.loc[index]
        rows.append({
            "feature_kind": "dam",
            "feature_id": dam_id_text(pd.Series([_text(row["AR_ID"])])).iloc[0],
            "action": ACTIONS[action],
            "nhdplus_id": str(int(pd.to_numeric(row["nhdplus_id"]))) if moved else "",
            "reason": reason.replace("\n", " "),
            "reviewed_by": reviewer,
            "reviewed_on": reviewed_on,
            "corrected_lat": f"{point.y:.7f}" if moved else "",
            "corrected_lon": f"{point.x:.7f}" if moved else "",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def merge_overrides(path, new):
    """(comment lines, merged rows, replaced feature ids)."""
    path = Path(path)
    comments, existing = [], pd.DataFrame(columns=COLUMNS)
    if path.exists():
        comments = [line for line in path.read_text().splitlines() if line.startswith("#")]
        existing = pd.read_csv(path, dtype=str, comment="#", keep_default_na=False)
    for column in COLUMNS:
        if column not in existing:
            existing[column] = ""
    same = existing["feature_kind"].eq("dam") & existing["feature_id"].isin(set(new["feature_id"]))
    replaced = sorted(existing.loc[same, "feature_id"])
    merged = pd.concat([existing.loc[~same, COLUMNS], new[COLUMNS]], ignore_index=True)
    return comments, merged, replaced


def write_overrides(path, comments, rows):
    """Write, validate with snapping.load_overrides, then replace."""
    path = Path(path)
    tmp = path.with_suffix(".csv.tmp")
    text = "\n".join(comments) + ("\n" if comments else "") + rows.to_csv(index=False)
    tmp.write_text(text)
    try:
        sn.load_overrides(tmp)
    except ValueError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", type=Path, required=True, help="The edited review GeoPackage.")
    parser.add_argument("--reviewer", required=True, help="Initials for reviewed_by.")
    parser.add_argument("--reviewed-on", default=date.today().isoformat())
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--dry-run", action="store_true", help="Check and show; write nothing.")
    args = parser.parse_args()

    corrections, dam_ids, flowlines = read_review(args.review)
    if corrections.empty:
        print("The corrections layer is empty: nothing to do.")
        return
    errors = check_corrections(corrections, dam_ids, flowlines)
    if errors:
        print(f"{len(errors)} problem(s); nothing written:\n  - " + "\n  - ".join(errors))
        sys.exit(1)

    new = override_rows(corrections, args.reviewer, args.reviewed_on)
    comments, merged, replaced = merge_overrides(args.overrides, new)
    counts = new["action"].value_counts().to_dict()
    print(f"{len(new)} correction(s): {counts.get('accept', 0)} moved, {counts.get('reject', 0)} rejected")
    if replaced:
        print(f"Replacing earlier overrides for: {replaced}")
    if args.dry_run:
        print(new.to_string(index=False))
        print("Dry run: nothing written.")
        return
    write_overrides(args.overrides, comments, merged)
    print(f"Wrote {args.overrides} ({len(merged)} override rows).")


if __name__ == "__main__":
    main()
