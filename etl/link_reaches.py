"""
Stage 5 of the reach-matching rework: link dams to stations along the
NHDPlus HR network using the stage-3 station reaches and stage-4 dam
reaches, and compare the result with the current links
(link_dams_stations.py). Rules in etl/common/reach_links.py.

Reads data/processed/station_reaches/<case_study>/ (select_station_reaches.py),
data/processed/dam_reaches/<case_study>/ (match_dams.py), the NHDPlus
packages and the current data/processed/dam_station_links_<huc8>.csv, and
writes data/processed/reach_links/<case_study>/ (--output-dir):

    reach_links_<huc8>.parquet      every connected (dam, station) pair whose
        endpoints both have a reach (accepted, or ambiguous with its
        provisional reach): relationship, network_distance_km, both
        endpoints' status and reason, usable (both accepted; the only pairs
        for analysis), station_in_mapped_reservoir (the station's reach is a
        path of the dam's reservoir). Unconnected pairs are not written.
    link_comparison_<huc8>.parquet  every pair in the old or new exposure
        window (config `exposure`): change, blocker, old / new relationship
        and distance
    summary.csv                     per HUC8: pairs by change, usable links,
                                    digitization check

The current links and every later stage are not touched; the summary at the
end of the log is for deciding whether the new reaches replace the old ones.

    python etl/link_reaches.py [--huc8 ...] [--hu4 ...] [--max-km KM]
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from etl.common import dam_matching as dm
from etl.common import reach_links as rl
from etl.common import water_features as wf
from etl.common.config import default_case_study, load_config
from etl.common.network import build_reach_graph
from etl.common.nhdplus import find_huc8_geodatabase
from etl.snap_features import BBOX_MARGIN_DEG, DAM_ID, STATION_ID
from etl.station_candidates import read_layer


def dam_id_text(series):
    """AR_IDs as text the way the old link tables write them ("123", not
    "123.0")."""
    numeric = pd.to_numeric(series, errors="coerce")
    whole = numeric.notna() & (numeric == numeric.round())
    out = series.astype(str)
    out[whole] = numeric[whole].astype("int64").astype(str)
    return out


class Networks:
    """Directed reach graph per NHDPlus package, kept while consecutive
    HUC8s share it (an HU4 fallback package serves several)."""

    def __init__(self):
        self.path, self.value = None, None

    def get(self, gdb_path):
        if gdb_path != self.path:
            streams = read_layer(gdb_path, "NHDFlowline", ["NHDPlusID", "LengthKM"],
                                 geometry=False)
            flow = read_layer(gdb_path, "NHDPlusFlow", ["FromNHDPID", "ToNHDPID"],
                              geometry=False)
            graph, lengths, stats = build_reach_graph(pd.DataFrame(streams), pd.DataFrame(flow))
            self.path, self.value = gdb_path, (graph, lengths, stats, dm.flow_edges(flow))
        return self.value


def read_lines(gdb_path, frames):
    """Flowline geometry near the stations and dams, indexed by NHDPlusID
    (package CRS)."""
    lon = pd.concat([frames[0]["LongitudeMeasure"], frames[1]["Longitude"]]).astype(float)
    lat = pd.concat([frames[0]["LatitudeMeasure"], frames[1]["Latitude"]]).astype(float)
    bbox = (lon.min() - BBOX_MARGIN_DEG, lat.min() - BBOX_MARGIN_DEG,
            lon.max() + BBOX_MARGIN_DEG, lat.max() + BBOX_MARGIN_DEG)
    lines = read_layer(gdb_path, "NHDFlowline", ["NHDPlusID"], bbox)
    lines["NHDPlusID"] = wf.clean_nhdplus_id(lines["NHDPlusID"])
    return lines.dropna(subset=["NHDPlusID"]).drop_duplicates("NHDPlusID") \
        .set_index("NHDPlusID")


def with_provisional_fractions(frame, lines):
    """frame plus provisional_fraction (its reported point projected onto
    the provisional reach, in the frame's metric CRS)."""
    frame = frame.copy()
    frame["provisional_fraction"] = np.nan
    crs = frame.get("metric_crs", pd.Series(dtype=object)).dropna()
    if crs.empty or lines is None:
        return frame
    frame["provisional_fraction"] = rl.provisional_fractions(frame, lines.to_crs(crs.iat[0]))
    return frame


def impounded_paths(dams):
    if "impounded_nhdplus_ids" not in dams:
        return {}
    return {dam_id: set(dm._split_ids(ids))
            for dam_id, ids in zip(dams["dam_id"], dams["impounded_nhdplus_ids"])}


def process_huc8(huc8, station_dir, dam_dir, processed, networks, limits, max_km):
    station_path = station_dir / f"station_reaches_{huc8}.parquet"
    dam_path = dam_dir / f"dam_reaches_{huc8}.parquet"
    stations = pd.read_parquet(station_path) if station_path.exists() else None
    dams = pd.read_parquet(dam_path) if dam_path.exists() else None
    if stations is None or dams is None or stations.empty or dams.empty:
        return None
    stations[STATION_ID] = stations[STATION_ID].astype(str)
    dams["dam_id"] = dam_id_text(dams[DAM_ID])

    has_reach = dams["match_status"].ne("rejected").any() and \
        stations["snap_status"].ne("rejected").any()
    try:
        gdb_path = find_huc8_geodatabase(processed / "nhdplus_extracted", huc8)
    except FileNotFoundError:
        gdb_path = None
    lines = read_lines(gdb_path, (stations, dams)) if gdb_path is not None and has_reach else None
    stations = with_provisional_fractions(stations, lines)
    dams = with_provisional_fractions(dams, lines)
    station_ends = rl.endpoints(stations, STATION_ID, "snap_status", "snap_reason")
    dam_ends = rl.endpoints(dams, "dam_id", "match_status", "match_reason")

    digitized = pd.Series(dtype=bool)
    if lines is not None:
        graph, lengths, stats, edges = networks.get(gdb_path)
        links = rl.link_pairs(dam_ends, station_ends, graph, lengths, max_km=max_km,
                              impounded=impounded_paths(dams))
        used = pd.concat([dam_ends["reach_id"], station_ends["reach_id"]]).dropna().unique()
        digitized = rl.digitized_downstream(lines.to_crs(stations["metric_crs"].dropna().iat[0]),
                                            edges, used)
    else:
        links = pd.DataFrame(columns=rl.LINK_COLUMNS)
    links.insert(0, "huc8", huc8)

    old_path = processed / f"dam_station_links_{huc8}.csv"
    if old_path.exists():
        old = pd.read_csv(old_path, dtype={"dam_id": str, "station_id": str},
                          usecols=["dam_id", "station_id", "relationship", "network_distance_km"])
        old["dam_id"] = dam_id_text(old["dam_id"])
        comparison = rl.compare_links(old, links, dam_ends, station_ends, limits)
        comparison.insert(0, "huc8", huc8)
    else:
        comparison = None
    return links, comparison, digitized


def print_summary(links, comparison, digitized, limits):
    usable = links[links["usable"]]
    window = rl.in_window(links["relationship"], links["network_distance_km"], limits)
    print("\n=== New links (usable = dam and station both accepted) ===")
    print(pd.crosstab(links["relationship"], links["usable"].map({True: "usable",
                                                                    False: "not_usable"}),
                      margins=True).to_string())
    print(f"usable links in the exposure window {limits}: {int((window & links['usable']).sum()):,}")
    up = usable[usable["relationship"] == "upstream"]
    print(f"usable upstream links with the station in the dam's mapped reservoir: "
          f"{int(up['station_in_mapped_reservoir'].sum()):,} of {len(up):,}")
    print(f"distance missing on usable links: {int(usable['network_distance_km'].isna().sum()):,}")

    if digitized.size:
        print(f"\nflowlines digitized downstream (last vertex nearer the flowline below): "
              f"{int(digitized.sum()):,} of {digitized.size:,} dam/station reaches "
              f"({digitized.mean():.1%})")

    if comparison is None or comparison.empty:
        print("\nNo old link tables to compare with.")
        return
    print("\n=== Old vs new, pairs in either exposure window ===")
    counts = comparison["change"].value_counts().reindex(rl.CHANGES, fill_value=0)
    print(counts.to_string())
    print(f"old window: {int(comparison['old_in_window'].sum()):,} pairs; new usable window: "
          f"{int((comparison['new_in_window'] & comparison['usable']).sum()):,} pairs")

    blocked = comparison[comparison["change"].isin(["now_blocked", "blocked"])]
    if len(blocked):
        print("\n=== Pairs in a window but not usable, by blocker ===")
        table = blocked.groupby("blocker").agg(
            pairs=("station_id", "size"),
            in_old_window=("old_in_window", "sum"),
            stations=("station_id", "nunique"),
            dams=("dam_id", "nunique")).sort_values("pairs", ascending=False)
        print(table.to_string())

    kept = comparison[comparison["change"] == "kept"]
    if len(kept):
        change = kept["distance_change_km"].abs()
        print("\n=== Kept pairs ===")
        print(f"relationship changed: {int(kept['relationship_changed'].sum()):,} of {len(kept):,}")
        print(f"|distance change| (km): {change.describe().round(3).to_dict()}")
        print(f"moved more than 1 km: {int((change > 1).sum()):,}")

    print("\n=== Dams with stations in the window ===")
    rows = {}
    for label, frame, flag in (("old", comparison, "old_in_window"),
                               ("new", comparison[comparison["usable"]], "new_in_window")):
        rel = "old_relationship" if label == "old" else "new_relationship"
        inside = frame[frame[flag]]
        sides = inside.groupby("dam_id")[rel].agg(set)
        rows[label] = {
            "any": len(sides),
            "downstream": int(sides.map(lambda s: "downstream" in s).sum()),
            "upstream": int(sides.map(lambda s: "upstream" in s).sum()),
            "both": int(sides.map(lambda s: {"upstream", "downstream"} <= s).sum()),
        }
    print(pd.DataFrame(rows).to_string())


def huc8_counts(huc8, links, comparison, digitized):
    row = {"huc8": huc8, "pairs": len(links), "usable": int(links["usable"].sum()),
           "digitized_checked": digitized.size, "digitized_downstream": int(digitized.sum())}
    if comparison is not None:
        row.update(comparison["change"].value_counts().reindex(rl.CHANGES, fill_value=0)
                   .to_dict())
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--huc8", nargs="+", help="Only these HUC8s.")
    parser.add_argument("--hu4", nargs="+", help="Only HUC8s in these HUC4s.")
    parser.add_argument("--station-dir", type=Path)
    parser.add_argument("--dam-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-km", type=float, default=None,
                        help="Stop each network search this far from the dam (default: no "
                             "limit, as the old links; a limit makes pairs beyond it look "
                             "unconnected in the comparison).")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    processed = cfg.project_root / "data" / "processed"
    station_dir = args.station_dir or processed / "station_reaches" / cfg.name
    dam_dir = args.dam_dir or processed / "dam_reaches" / cfg.name
    output_dir = args.output_dir or processed / "reach_links" / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    exposure = cfg.get("exposure", {}) or {}
    limits = {"max_downstream_km": exposure.get("max_downstream_km"),
              "max_upstream_km": exposure.get("max_upstream_km")}

    huc8s = sorted(p.stem.removeprefix("station_reaches_")
                   for p in station_dir.glob("station_reaches_*.parquet"))
    if args.huc8:
        huc8s = [h for h in huc8s if h in set(args.huc8)]
    if args.hu4:
        huc8s = [h for h in huc8s if h[:4] in set(args.hu4)]
    print(f"Case study: {cfg.name}; HUC8s with station reaches: {len(huc8s)}; "
          f"exposure window: {limits}; max_km: {args.max_km}", flush=True)

    networks = Networks()
    all_links, all_comparisons, all_digitized, counts, failed = [], [], [], [], []
    for number, huc8 in enumerate(huc8s, start=1):
        try:
            result = process_huc8(huc8, station_dir, dam_dir, processed, networks, limits,
                                  args.max_km)
        except Exception as exc:
            print(f"[{number}/{len(huc8s)}] HUC8 {huc8} FAILED: {exc}", flush=True)
            failed.append(huc8)
            continue
        if result is None:
            print(f"[{number}/{len(huc8s)}] HUC8 {huc8}: no dams or no stations", flush=True)
            continue
        links, comparison, digitized = result
        links.to_parquet(output_dir / f"reach_links_{huc8}.parquet", index=False)
        if comparison is not None:
            comparison.to_parquet(output_dir / f"link_comparison_{huc8}.parquet", index=False)
            all_comparisons.append(comparison)
        all_links.append(links)
        all_digitized.append(digitized)
        counts.append(huc8_counts(huc8, links, comparison, digitized))
        old_text = "no old links" if comparison is None else \
            comparison["change"].value_counts().to_dict()
        print(f"[{number}/{len(huc8s)}] HUC8 {huc8}: {len(links):,} pairs, "
              f"{int(links['usable'].sum()):,} usable; {old_text}", flush=True)

    if counts:
        pd.DataFrame(counts).to_csv(output_dir / "summary.csv", index=False)
        print_summary(pd.concat(all_links, ignore_index=True),
                      pd.concat(all_comparisons, ignore_index=True) if all_comparisons else None,
                      pd.concat(all_digitized), limits)
    print(f"\nSaved to: {output_dir}")
    if failed:
        raise RuntimeError(f"{len(failed)} HUC8(s) failed: {failed}")


if __name__ == "__main__":
    main()
