"""
Regression check of stage 3 (select_station_reaches.py) against stations
with a known reach problem (config/known_station_reaches.csv).

Two separate checks per station, reported in separate columns:

    avoids_wrong_reach   pass: the station is not accepted on the reach the
                         old snapping wrongly chose (ambiguous, rejected or
                         another reach all pass); fail: accepted on it;
                         unknown: no wrong reach recorded
    correct_reach        confirmed: accepted on the reviewed correct reach
                         (correct_nhdplus_id, or correct_river_name matching
                         the reach's GNIS name); not_confirmed: a reviewed
                         answer exists and the station isn't accepted on it;
                         not_reviewed: no reviewed answer yet

Passing avoids_wrong_reach is NOT evidence of a correct match: a station
can avoid the wrong reach and still be on another wrong one. Only
correct_reach = confirmed says the match is right, and only for stations
someone has reviewed.

The run fails (exit 1) if any station is accepted on its known wrong
reach, or is not_confirmed against a reviewed answer.

    python etl/check_known_stations.py                       # check stage-3 outputs
    python etl/check_known_stations.py --fill-wrong-reaches  # record the old
        wrong reaches (then commit the CSV, so later reruns of the old
        snapping can't change them)

A station may have several wrong reaches, one row each, from:
    data/processed/snapped_features_<huc8>.gpkg (stations_snapped): the
        assignment the audit map reviewed (nearest reach per station, as
        compare_snapping.py reads it);
    snap_diagnostics_<huc8>.csv in data/processed/ and
        data/processed/snap_validation/: the Oct 2026 scored snapping's
        choice, which may differ.

Writes known_stations_check.csv next to the stage-3 outputs.
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common import snapping as sn
from etl.common.config import default_case_study, load_config

KNOWN_FILE = Path("config") / "known_station_reaches.csv"
STATION_ID = "MonitoringLocationIdentifier"


def load_known(path):
    return pd.read_csv(path, comment="#", dtype="string").apply(lambda s: s.str.strip())


def wrong_reaches_from_gpkgs(paths, station_ids):
    """(station_id, nhdplus_id, source) from old snapped_features files."""
    from etl.compare_snapping import old_assignments

    rows = []
    for path in paths:
        old = old_assignments(path)
        old = old[(old["feature_kind"] == "station") & old["feature_id"].isin(station_ids)
                  & old["old_reach"].notna()]
        rows += [(sid, int(reach), str(path)) for sid, reach in zip(old["feature_id"], old["old_reach"])]
    return rows


def wrong_reaches_from_diagnostics(paths, station_ids):
    """(station_id, nhdplus_id, source) from snap_diagnostics files."""
    rows = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"feature_id": str}, low_memory=False,
                            usecols=["feature_kind", "feature_id", "NHDPlusID"])
        frame = frame[(frame["feature_kind"] == "station") & frame["feature_id"].isin(station_ids)
                      & frame["NHDPlusID"].notna()]
        rows += [(sid, int(reach), str(path)) for sid, reach in zip(frame["feature_id"],
                                                                     frame["NHDPlusID"])]
    return rows


def fill_wrong_reaches(known, found):
    """Add a row for every (station, wrong reach) in `found` (tuples of
    station_id, nhdplus_id, source) not already recorded; a station's
    empty placeholder row is replaced. Reviewed columns are copied."""
    known = known.copy()
    have = set(zip(known["station_id"], known["wrong_nhdplus_id"].dropna()))
    reviewed = ["correct_nhdplus_id", "correct_river_name", "reviewed_by", "reviewed_on", "notes"]
    first = known.drop_duplicates("station_id").set_index("station_id")[reviewed]
    new = []
    for station, reach, source in found:
        if (station, str(reach)) in have:
            continue
        have.add((station, str(reach)))
        new.append({"station_id": station, "wrong_nhdplus_id": str(reach),
                    "wrong_source": source, **first.loc[station].to_dict()})
    if not new:
        return known
    filled = pd.concat([known, pd.DataFrame(new)], ignore_index=True).astype("string")
    has_reach = filled.groupby("station_id")["wrong_nhdplus_id"].transform(
        lambda s: s.notna().any())
    filled = filled[filled["wrong_nhdplus_id"].notna() | ~has_reach]
    order = {sid: i for i, sid in enumerate(known["station_id"].drop_duplicates())}
    return filled.sort_values(["station_id", "wrong_source"], key=lambda s: s.map(order)
                              if s.name == "station_id" else s).reset_index(drop=True)


def check(known, reaches):
    """One row per known station: its stage-3 outcome and the two checks.
    The station fails avoids_wrong_reach if accepted on any of its rows'
    wrong reaches."""
    reaches = reaches.drop_duplicates(STATION_ID).set_index(STATION_ID)
    rows = []
    for station, group in known.groupby("station_id", sort=False):
        wrong = {int(r) for r in group["wrong_nhdplus_id"].dropna()}
        correct_id = group["correct_nhdplus_id"].dropna()
        correct_name = group["correct_river_name"].dropna()
        out = {"station_id": station,
               "wrong_nhdplus_ids": ";".join(str(r) for r in sorted(wrong)) or None,
               "correct_nhdplus_id": correct_id.iat[0] if len(correct_id) else None,
               "correct_river_name": correct_name.iat[0] if len(correct_name) else None}
        reviewed = len(correct_id) or len(correct_name)
        if station not in reaches.index:
            out.update(found=False, avoids_wrong_reach="unknown",
                       correct_reach="not_confirmed" if reviewed else "not_reviewed")
            rows.append(out)
            continue
        result = reaches.loc[station]
        accepted = result["snap_status"] == sn.ACCEPTED
        reach = None if pd.isna(result.get("NHDPlusID")) else int(result["NHDPlusID"])
        out.update(found=True, snap_status=result["snap_status"],
                   snap_reason=result["snap_reason"], nhdplus_id=reach,
                   provisional_nhdplus_id=result.get("provisional_nhdplus_id"),
                   reach_gnis_name=result.get("reach_gnis_name"))

        if not wrong:
            out["avoids_wrong_reach"] = "unknown"
        else:
            out["avoids_wrong_reach"] = "fail" if accepted and reach in wrong else "pass"

        if len(correct_id):
            right = accepted and reach == int(correct_id.iat[0])
        elif len(correct_name):
            right = accepted and sn.name_matches(correct_name.iat[0],
                                                 result.get("reach_gnis_name")) is True
        if reviewed:
            out["correct_reach"] = "confirmed" if right else "not_confirmed"
        else:
            out["correct_reach"] = "not_reviewed"
        rows.append(out)
    return pd.DataFrame(rows)


def report(result):
    """Printed summary; returns True if the regression check passes."""
    avoid = result["avoids_wrong_reach"].value_counts()
    correct = result["correct_reach"].value_counts()
    print(result.to_string(index=False))
    print(f"\nAvoided the known wrong reach: {avoid.get('pass', 0)} pass, "
          f"{avoid.get('fail', 0)} fail, {avoid.get('unknown', 0)} unknown "
          f"(of {len(result)})")
    print(f"Correct reach: {correct.get('confirmed', 0)} confirmed, "
          f"{correct.get('not_confirmed', 0)} not confirmed, "
          f"{correct.get('not_reviewed', 0)} not reviewed")
    print("NOTE: avoiding a known wrong reach does not show the station is matched "
          "correctly; only 'confirmed' (against a reviewed reach) does.")
    return avoid.get("fail", 0) == 0 and correct.get("not_confirmed", 0) == 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--reach-dir", type=Path)
    parser.add_argument("--known", type=Path)
    parser.add_argument("--fill-wrong-reaches", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    processed = cfg.project_root / "data" / "processed"
    known_path = args.known or cfg.project_root / KNOWN_FILE
    known = load_known(known_path)

    if args.fill_wrong_reaches:
        ids = set(known["station_id"])
        gpkgs = sorted(processed.glob("snapped_features_*.gpkg"))
        diagnostics = (sorted(processed.glob("snap_diagnostics_*.csv"))
                       + sorted((processed / "snap_validation").glob("snap_diagnostics_*.csv")))
        found = (wrong_reaches_from_gpkgs(gpkgs, ids)
                 + wrong_reaches_from_diagnostics(diagnostics, ids))
        filled = fill_wrong_reaches(known, found)
        header = [line for line in known_path.read_text().splitlines() if line.startswith("#")]
        known_path.write_text("\n".join(header) + "\n" + filled.to_csv(index=False))
        print(filled[["station_id", "wrong_nhdplus_id", "wrong_source"]].to_string(index=False))
        missing = sorted(ids - set(filled.dropna(subset=["wrong_nhdplus_id"])["station_id"]))
        print(f"\nRead {len(gpkgs)} snapped_features and {len(diagnostics)} diagnostics "
              f"file(s). No wrong reach found for: {missing or 'none'}")
        reaches = filled.dropna(subset=["wrong_nhdplus_id"]).groupby("station_id")
        disagree = sorted(station for station, g in reaches
                          if g["wrong_nhdplus_id"].nunique() > 1)
        if disagree:
            print(f"CHECK BEFORE COMMITTING: the sources give different reaches for "
                  f"{disagree}. The later snapping may have fixed some of them; delete "
                  "any row whose reach is actually right.")
        print(f"Updated {known_path}; review and commit it.")
        return

    reach_dir = args.reach_dir or processed / "station_reaches" / cfg.name
    files = sorted(reach_dir.glob("station_reaches_*.parquet"))
    reaches = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files \
        else pd.DataFrame(columns=[STATION_ID, "snap_status", "snap_reason", "NHDPlusID"])
    reaches[STATION_ID] = reaches[STATION_ID].astype(str)
    result = check(known, reaches)
    result.to_csv(reach_dir / "known_stations_check.csv", index=False)
    if not report(result):
        sys.exit(1)


if __name__ == "__main__":
    main()
