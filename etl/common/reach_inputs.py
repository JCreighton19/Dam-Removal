"""The evidence-based reach matching (stages 3-5) as warehouse inputs.

load_core_tables.py and download_wqp_results.py were written against the
nearest-flowline outputs (snapped_features_<huc8>.gpkg from snap_features.py,
dam_station_links_<huc8>.csv from link_dams_stations.py). This module reads
the stage 3-5 outputs instead and returns frames with the same columns, so
the table builders and the WQP download don't change:

    data/processed/dam_reaches/<case_study>/dam_reaches_<huc8>.parquet         (match_dams.py)
    data/processed/station_reaches/<case_study>/station_reaches_<huc8>.parquet (select_station_reaches.py)
    data/processed/station_candidates/<case_study>/station_candidates_<huc8>.parquet
    data/processed/reach_links/<case_study>/reach_links_<huc8>.parquet         (link_reaches.py)

What is kept:

    links     usable pairs only (both endpoints accepted, or a
              confluence-ambiguous station whose every candidate reach
              links the same way: usable_reason robust_to_reach_choice)
    dams      accepted dams (an ambiguous dam has no reach to link from)
    stations  stations in a usable link. A robust_to_reach_choice station
              is placed on its provisional reach (the one its links were
              measured from), match_method "provisional_reach", and keeps
              its reported location as geometry, since which reach it is
              on is not settled.

The config key `link_source` picks the source: "reach_links" (default) or
"snapped" (the old snap_features / link_dams files, for case studies that
never ran stages 1-5).
"""

from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from etl.common import reach_selection as rs
from etl.common import snapping as sn
from etl.common.nhdplus import find_huc8_geodatabase, read_nhdplus_layer
from etl.common.progress import HucProgress

REACH_LINKS, SNAPPED = "reach_links", "snapped"
LINK_SOURCES = (REACH_LINKS, SNAPPED)

STATION_ID = "MonitoringLocationIdentifier"
DAM_ID = "AR_ID"
NHDPLUS_CRS = "EPSG:4269"
WGS84 = "EPSG:4326"
LINKED = ("upstream", "downstream", "same_reach")
PROVISIONAL = "provisional_reach"

# Columns the dam / station stage outputs carry that come from the source
# files (dam CSV, WQP station selection) instead, so they aren't duplicated.
DAM_STAGE_COLUMNS = ["huc8", "NHDPlusID", "snap_distance_m", "snap_fraction", "snapped_lon",
                     "snapped_lat", "match_method", "match_reason", "match_evidence",
                     "datum_epsg", "datum_assumed"]
STATION_STAGE_COLUMNS = ["huc8", STATION_ID, "snap_status", "snap_reason", "snap_evidence",
                         "match_method", "snap_distance_m", "snapped_lon", "snapped_lat",
                         "x_m", "y_m", "metric_crs", "datum_epsg", "datum_assumed"]


def link_source(cfg):
    source = cfg.get("link_source", REACH_LINKS)
    if source not in LINK_SOURCES:
        raise ValueError(f"Config link_source is {source!r}; expected one of {LINK_SOURCES}")
    return source


def stage_dir(cfg, stage):
    return cfg.project_root / "data" / "processed" / stage / cfg.name


def dam_id_text(series):
    """AR_IDs as text the way the old link tables write them ("123", not
    "123.0")."""
    numeric = pd.to_numeric(series, errors="coerce")
    whole = numeric.notna() & (numeric == numeric.round())
    out = series.astype(str).str.strip()
    out[whole] = numeric[whole].astype("int64").astype(str)
    return out


def read_per_huc8(cfg, stage, prefix, huc8s):
    """Concatenate <stage>/<case_study>/<prefix>_<huc8>.parquet over huc8s
    (HUC8s without a file are left out: no dams, no stations or no package)."""
    folder = stage_dir(cfg, stage)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} does not exist. Run the reach-matching stages (station_candidates, "
            "select_station_reaches, match_dams, link_reaches) for this case study, or set "
            f"link_source: {SNAPPED} in the config to use the snap_features / link_dams files.")
    # A HUC8 without a file is only "no dams / no stations / no package" once
    # check_coverage has confirmed the stages ran on it.
    frames = [pd.read_parquet(path) for huc8 in huc8s
              if (path := folder / f"{prefix}_{huc8}.parquet").exists()]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "huc8" in out:
        out["huc8"] = out["huc8"].astype(str).str.zfill(8)
    return out


# --------------------------------------------------
# Links
# --------------------------------------------------

def usable_links(links, dams):
    """Usable, connected pairs in the old link-table shape (dam_id,
    dam_name, station_id, year_removed, relationship, network_distance_km,
    distance_method, dam_reach_id, station_reach_id), plus huc8,
    station_fraction and usable_reason. `dams` is the dam_reaches table
    (AR_ID, Dam_Name, Year_Removed)."""
    columns = ["huc8", "dam_id", "dam_name", "station_id", "year_removed", "relationship",
               "network_distance_km", "distance_method", "dam_reach_id", "station_reach_id",
               "station_fraction", "usable_reason"]
    if links.empty:
        return pd.DataFrame(columns=columns)
    keep = links["usable"].astype("boolean").fillna(False).astype(bool) \
        & links["relationship"].isin(LINKED)
    out = links.loc[keep].copy()
    out["dam_id"] = dam_id_text(out["dam_id"])
    out["station_id"] = out["station_id"].astype(str)
    names = pd.DataFrame({"dam_id": dam_id_text(dams[DAM_ID]),
                          "dam_name": dams.get("Dam_Name"),
                          "year_removed": dams.get("Year_Removed")}).drop_duplicates("dam_id")
    out = out.merge(names, on="dam_id", how="left")
    return out[columns].reset_index(drop=True)


def check_coverage(cfg, huc8s):
    """Every HUC8 must have been processed by link_reaches.py (its progress
    marker), since a HUC8 with no reach_links file is otherwise read as "no
    links". Stops on HUC8s the stages never ran on, e.g. after a run limited
    to a few HUC4s."""
    progress = HucProgress(cfg, "link_reaches")
    missing = sorted(huc8 for huc8 in huc8s if not progress.done(huc8))
    if missing:
        hu4s = sorted({huc8[:4] for huc8 in missing})
        raise ValueError(
            f"{len(missing)} of {len(huc8s)} HUC8(s) were never processed by the reach-matching "
            f"stages (no link_reaches marker in {progress.folder}), e.g. {missing[:10]}; "
            f"HU4s: {hu4s[:20]}{' ...' if len(hu4s) > 20 else ''}. Run station_candidates, "
            "select_station_reaches, match_dams and link_reaches for every HUC8 "
            "(bash data/slurm/run_pipeline.sh --from download_streams --skip "
            f"snap_features,link_dams), or set link_source: {SNAPPED} in the config.")


def load_usable_links(cfg, huc8s):
    links = read_per_huc8(cfg, "reach_links", "reach_links", huc8s)
    check_coverage(cfg, huc8s)
    dams = read_per_huc8(cfg, "dam_reaches", "dam_reaches", huc8s)
    if dams.empty:
        dams = pd.DataFrame(columns=[DAM_ID, "Dam_Name", "Year_Removed"])
    out = usable_links(links, dams)
    reasons = out["usable_reason"].value_counts().to_dict()
    print(f"Reach links: {len(links):,} connected pairs, {len(out):,} usable {reasons}",
          flush=True)
    return out


# --------------------------------------------------
# Reach attributes (NHDPlus package)
# --------------------------------------------------

def reach_attributes(nhd_dir, reach_ids_by_huc8):
    """reach_ftype, reach_fcode, reach_gnis_name, reach_totdasqkm and
    reach_slope by NHDPlusID, read from each HUC8's package (NHDFlowline +
    NHDPlusFlowlineVAA, placeholders as NaN). Packages shared by several
    HUC8s are read once."""
    by_gdb = defaultdict(set)
    for huc8, ids in reach_ids_by_huc8.items():
        ids = {int(i) for i in pd.Series(list(ids)).dropna()}
        if ids:
            by_gdb[find_huc8_geodatabase(nhd_dir, huc8)] |= ids

    frames = []
    for gdb_path, ids in by_gdb.items():
        lines = read_nhdplus_layer(gdb_path, "NHDFlowline",
                                   columns=["NHDPlusID", "GNIS_Name", "FType", "FCode"],
                                   ignore_geometry=True)
        lines["NHDPlusID"] = pd.to_numeric(lines["NHDPlusID"], errors="coerce").round() \
            .astype("Int64")
        lines = lines[lines["NHDPlusID"].isin(ids)].drop_duplicates("NHDPlusID")
        layers = {str(name).lower() for name, _ in pyogrio.list_layers(gdb_path)}
        if "nhdplusflowlinevaa" in layers:
            vaa = read_nhdplus_layer(gdb_path, "NHDPlusFlowlineVAA",
                                     columns=["NHDPlusID", "TotDASqKm", "Slope"],
                                     ignore_geometry=True)
            vaa["NHDPlusID"] = pd.to_numeric(vaa["NHDPlusID"], errors="coerce").round() \
                .astype("Int64")
            vaa = vaa[vaa["NHDPlusID"].isin(ids)].drop_duplicates("NHDPlusID")
            lines = lines.merge(vaa, on="NHDPlusID", how="left")
        else:
            print(f"WARNING: {gdb_path.name} has no NHDPlusFlowlineVAA; no drainage areas "
                  "or slopes", flush=True)
            lines["TotDASqKm"], lines["Slope"] = np.nan, np.nan
        frames.append(pd.DataFrame(lines))

    columns = ["reach_ftype", "reach_fcode", "reach_gnis_name", "reach_totdasqkm",
               "reach_slope"]
    if not frames:
        return pd.DataFrame(columns=columns, index=pd.Index([], dtype="Int64", name="NHDPlusID"))
    lines = pd.concat(frames, ignore_index=True).drop_duplicates("NHDPlusID")
    return pd.DataFrame({
        "reach_ftype": pd.to_numeric(lines["FType"], errors="coerce").astype("Int16").to_numpy(),
        "reach_fcode": pd.to_numeric(lines["FCode"], errors="coerce").astype("Int32").to_numpy(),
        "reach_gnis_name": lines["GNIS_Name"].to_numpy(),
        "reach_totdasqkm": sn.valid_area(lines["TotDASqKm"]).to_numpy(),
        "reach_slope": sn.valid_slope(lines["Slope"]).to_numpy(),
    }, index=pd.Index(lines["NHDPlusID"], name="NHDPlusID"))


def _ids_by_huc8(frame):
    return {huc8: set(group["NHDPlusID"].dropna()) for huc8, group in frame.groupby("huc8")}


def _join_attributes(frame, attributes):
    reach = frame["NHDPlusID"].astype("Int64")
    found = attributes.reindex(reach.to_numpy())
    found.index = frame.index
    return frame.join(found)


def _far_snap(evidence):
    return evidence.astype("string").fillna("").str.split(";") \
        .map(lambda codes: "far_snap" in codes).astype(bool)


# --------------------------------------------------
# Dams
# --------------------------------------------------

def dam_points(reaches, source, attributes):
    """Accepted dams shaped like the old dams_snapped layer: every dam CSV
    column (`source`), plus the reach and match columns from stage 4 and
    `attributes`; geometry = the dam projected onto its reach (WGS84)."""
    reaches = reaches.copy()
    reaches[DAM_ID] = dam_id_text(reaches[DAM_ID])
    accepted = reaches["match_status"].eq(rs.ACCEPTED) & reaches["NHDPlusID"].notna()
    if (~accepted).any():
        dropped = reaches.loc[~accepted].groupby(["match_status", "match_reason"],
                                                 dropna=False).size()
        print(f"Dams: left out {int((~accepted).sum()):,} dam(s) without an accepted reach:\n"
              f"{dropped.to_string()}", flush=True)
    reaches = reaches.loc[accepted, [DAM_ID] + DAM_STAGE_COLUMNS]

    source = pd.DataFrame(source).drop(columns="geometry", errors="ignore").copy()
    source[DAM_ID] = dam_id_text(source[DAM_ID])
    source = source.drop_duplicates(DAM_ID)
    missing = sorted(set(reaches[DAM_ID]) - set(source[DAM_ID]))
    if missing:
        raise ValueError(f"Dam(s) in dam_reaches but not in the dam source: {missing}")
    source = source.drop(columns=[c for c in source if c in reaches and c != DAM_ID])
    out = reaches.merge(source, on=DAM_ID, how="left")

    out["NHDPlusID"] = out["NHDPlusID"].astype("Int64")
    out = _join_attributes(out, attributes)
    out = out.rename(columns={"match_reason": "snap_reason", "match_evidence": "snap_evidence",
                              "huc8": "pipeline_huc8"})
    out["qc_far_snap"] = _far_snap(out["snap_evidence"])
    # A nearest-flowline check; the evidence rules replace it.
    out["qc_near_unsuitable_flowline"] = False
    geometry = gpd.points_from_xy(out["snapped_lon"], out["snapped_lat"], crs=NHDPLUS_CRS)
    return gpd.GeoDataFrame(out, geometry=geometry, crs=NHDPLUS_CRS).to_crs(WGS84)


def load_dam_points(cfg, huc8s, dam_source):
    reaches = read_per_huc8(cfg, "dam_reaches", "dam_reaches", huc8s)
    if reaches.empty:
        raise ValueError(f"No dam_reaches files for the case study's HUC8s in "
                         f"{stage_dir(cfg, 'dam_reaches')}")
    attributes = reach_attributes(cfg.project_root / "data" / "processed" / "nhdplus_extracted",
                                  _ids_by_huc8(reaches[reaches["NHDPlusID"].notna()]))
    return dam_points(reaches, dam_source, attributes)


# --------------------------------------------------
# Stations
# --------------------------------------------------

def station_points(reaches, source, links, candidate_distances, attributes):
    """Stations in a usable link, shaped like the old stations_snapped
    layer: every WQP station column (`source`), the reach the links were
    measured from (NHDPlusID, snap_fraction), stage-3 match columns and
    `attributes`, reported_area_km2 and qc_non_riverine_name.

    snap_distance_m is stage 3's for accepted stations and, for a station
    on its provisional reach, the candidate distance to that reach
    (`candidate_distances`: huc8, station id, NHDPlusID, distance_m).
    Geometry (WGS84): the projected point when accepted, the reported point
    otherwise."""
    placed = links[["huc8", "station_id", "station_reach_id", "station_fraction"]] \
        .drop_duplicates(["huc8", "station_id", "station_reach_id"])
    split = placed.duplicated(["huc8", "station_id"], keep=False)
    if split.any():
        raise ValueError("Station(s) linked from more than one reach: "
                         f"{sorted(placed.loc[split, 'station_id'].unique())[:10]}")
    placed = placed.rename(columns={"station_id": STATION_ID, "station_reach_id": "NHDPlusID",
                                    "station_fraction": "snap_fraction"})

    reaches = reaches.copy()
    reaches[STATION_ID] = reaches[STATION_ID].astype(str)
    out = placed.merge(reaches[STATION_STAGE_COLUMNS], on=["huc8", STATION_ID], how="left")
    if out["snap_status"].isna().any():
        raise ValueError("Linked station(s) missing from station_reaches: "
                         f"{sorted(out.loc[out['snap_status'].isna(), STATION_ID])[:10]}")
    out["NHDPlusID"] = out["NHDPlusID"].astype("Int64")

    provisional = out["snap_status"].ne(rs.ACCEPTED)
    out.loc[provisional, "match_method"] = PROVISIONAL
    if provisional.any():
        distances = candidate_distances.assign(
            **{STATION_ID: candidate_distances[STATION_ID].astype(str),
               "NHDPlusID": candidate_distances["NHDPlusID"].astype("Int64")}) \
            .drop_duplicates(["huc8", STATION_ID, "NHDPlusID"])
        found = out.loc[provisional, ["huc8", STATION_ID, "NHDPlusID"]].merge(
            distances, on=["huc8", STATION_ID, "NHDPlusID"], how="left")
        out.loc[provisional, "snap_distance_m"] = found["distance_m"].to_numpy()

    station_source = pd.DataFrame(source).drop(columns="geometry", errors="ignore").copy()
    station_source[STATION_ID] = station_source[STATION_ID].astype(str)
    station_source = station_source.drop_duplicates(STATION_ID)
    station_source = station_source.drop(
        columns=[c for c in station_source if c in out and c != STATION_ID])
    missing = sorted(set(out[STATION_ID]) - set(station_source[STATION_ID]))
    if missing:
        raise ValueError(f"Linked station(s) not in the WQP station selection: {missing[:10]}")
    out = out.merge(station_source, on=STATION_ID, how="left")

    out = _join_attributes(out, attributes)
    out = out.rename(columns={"huc8": "pipeline_huc8"})
    out["reported_area_km2"] = sn.reported_area_km2(out).to_numpy()
    out["qc_non_riverine_name"] = sn.non_riverine_name(
        out.get("MonitoringLocationName", pd.Series(index=out.index, dtype="string"))).to_numpy()
    out["qc_far_snap"] = _far_snap(out["snap_evidence"])
    out["qc_near_unsuitable_flowline"] = False

    geometry = gpd.GeoSeries([None] * len(out), index=out.index, crs=WGS84)
    snapped = ~provisional & out["snapped_lon"].notna()
    if snapped.any():
        geometry[snapped] = gpd.GeoSeries(
            gpd.points_from_xy(out.loc[snapped, "snapped_lon"], out.loc[snapped, "snapped_lat"]),
            index=out.index[snapped], crs=NHDPLUS_CRS).to_crs(WGS84)
    reported = ~snapped & out["x_m"].notna()
    for crs, rows in out[reported].groupby("metric_crs"):
        geometry[rows.index] = gpd.GeoSeries(gpd.points_from_xy(rows["x_m"], rows["y_m"]),
                                             index=rows.index, crs=crs).to_crs(WGS84)
    return gpd.GeoDataFrame(out, geometry=geometry, crs=WGS84)


def load_station_points(cfg, huc8s, links, station_source):
    reaches = read_per_huc8(cfg, "station_reaches", "station_reaches", huc8s)
    linked_huc8s = sorted(links["huc8"].unique())
    if reaches.empty:
        reaches = pd.DataFrame(columns=STATION_STAGE_COLUMNS)
    reaches[STATION_ID] = reaches[STATION_ID].astype(str)

    provisional = links.merge(
        reaches.loc[reaches["snap_status"].ne(rs.ACCEPTED), ["huc8", STATION_ID]],
        left_on=["huc8", "station_id"], right_on=["huc8", STATION_ID])
    candidate_huc8s = sorted(provisional["huc8"].unique())
    candidates = read_per_huc8(cfg, "station_candidates", "station_candidates",
                               candidate_huc8s) if candidate_huc8s else pd.DataFrame()
    if candidates.empty:
        candidates = pd.DataFrame(columns=["huc8", STATION_ID, "NHDPlusID", "distance_m"])
    candidates = candidates[["huc8", STATION_ID, "NHDPlusID", "distance_m"]]

    reach_ids = links.rename(columns={"station_reach_id": "NHDPlusID"})[["huc8", "NHDPlusID"]]
    attributes = reach_attributes(cfg.project_root / "data" / "processed" / "nhdplus_extracted",
                                  _ids_by_huc8(reach_ids[reach_ids["huc8"].isin(linked_huc8s)]))
    return station_points(reaches, station_source, links, candidates, attributes)
