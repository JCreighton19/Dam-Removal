"""Mapped reservoir polygons of the dam removals (Stage 4, match_dams.py).

Used by etl/build_dam_reservoir.py (the warehouse table dam_reservoir) and
analysis/audits/reservoir_size_audit.py, so both read the polygons the same
way.

A dam has a mapped reservoir when Stage 4 put it at the main outlet of an
NHDWaterbody (reservoir_status dam_at_outlet). Its area is the polygon's
AreaSqKm attribute, else its geometry measured in EPSG:5070 (equal area).
These are measured values: they are never imputed, and a dam without a
polygon keeps a null area.

polygon_timing compares the polygon's FDate (the date it was last edited in
NHD) with the removal year:
    edited_by_removal_year  FDate year <= removal year: the polygon predates
                            the removal (or was last edited that year)
    edited_after_removal    may show a drained, shrunk or re-mapped reservoir
    date_unknown
FDate is an edit date, not a survey date, so edited_by_removal_year is the
safer subset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from etl.common.reach_inputs import dam_id_text

EQUAL_AREA_CRS = "EPSG:5070"
AT_OUTLET = "dam_at_outlet"
PRE_REMOVAL = "edited_by_removal_year"

DAM_REACH_COLUMNS = ["AR_ID", "huc8", "match_status", "reservoir_status", "reservoir_id",
                     "reservoir_ftype", "reservoir_fdate"]


def polygon_timing(fdate, year_removed):
    """edited_by_removal_year / edited_after_removal / date_unknown per row
    (fdate and year_removed are aligned Series)."""
    year = pd.to_datetime(fdate, errors="coerce").dt.year
    removed = pd.to_numeric(year_removed, errors="coerce")
    out = pd.Series("date_unknown", index=fdate.index, dtype="string")
    out[year.notna() & removed.notna() & (year <= removed)] = PRE_REMOVAL
    out[year.notna() & removed.notna() & (year > removed)] = "edited_after_removal"
    return out


def read_dam_reaches(folder):
    """Stage 4 dam_reaches_<huc8>.parquet files, one row per dam."""
    frames = []
    for path in sorted(Path(folder).glob("dam_reaches_*.parquet")):
        frame = pd.read_parquet(path)
        frames.append(frame[[c for c in DAM_REACH_COLUMNS if c in frame]])
    if not frames:
        return pd.DataFrame(columns=DAM_REACH_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["AR_ID"] = dam_id_text(out["AR_ID"])
    out["huc8"] = out["huc8"].astype(str).str.zfill(8)
    return out.drop_duplicates("AR_ID")


def reservoir_areas(reaches, nhd_dir):
    """polygon_area_km2 and polygon_area_source per reservoir_id, read from
    each HUC8's NHD package (attribute AreaSqKm, else geometry area).
    Returns (areas, HUC8s whose package wasn't found)."""
    import pyogrio

    from etl.common.nhdplus import find_huc8_geodatabase
    from etl.common.water_features import WATERBODY_LAYER, clean_permanent_id

    wanted = reaches.dropna(subset=["reservoir_id"])
    rows, missing_packages = [], set()
    by_package = {}
    for huc8, group in wanted.groupby("huc8"):
        try:
            path = find_huc8_geodatabase(nhd_dir, huc8)
        except FileNotFoundError:
            missing_packages.add(huc8)
            continue
        by_package.setdefault(path, set()).update(clean_permanent_id(group["reservoir_id"]))
    for path, ids in by_package.items():
        fields = {f.lower(): f for f in pyogrio.read_info(path, layer=WATERBODY_LAYER)["fields"]}
        columns = [fields["permanent_identifier"]] + (
            [fields["areasqkm"]] if "areasqkm" in fields else [])
        table = pyogrio.read_dataframe(path, layer=WATERBODY_LAYER, columns=columns,
                                       read_geometry=False, fid_as_index=True)
        table["reservoir_id"] = clean_permanent_id(table[fields["permanent_identifier"]])
        table = table[table["reservoir_id"].isin(ids)]
        if "areasqkm" in fields:
            area = pd.to_numeric(table[fields["areasqkm"]], errors="coerce")
            source = pd.Series("AreaSqKm", index=table.index)
        else:
            area = pd.Series(np.nan, index=table.index)
            source = pd.Series("geometry_5070", index=table.index)
        need_geometry = area.isna() | (area <= 0)
        if need_geometry.any():
            shapes = pyogrio.read_dataframe(path, layer=WATERBODY_LAYER, columns=[],
                                            fids=table.index[need_geometry].to_numpy())
            measured = shapes.to_crs(EQUAL_AREA_CRS).area.to_numpy() / 1e6
            area.loc[need_geometry] = measured
            source.loc[need_geometry] = "geometry_5070"
        rows.append(pd.DataFrame({"reservoir_id": table["reservoir_id"].to_numpy(),
                                  "polygon_area_km2": area.where(area > 0).to_numpy(),
                                  "polygon_area_source": source.to_numpy()}))
    out = (pd.concat(rows, ignore_index=True).drop_duplicates("reservoir_id")
           if rows else pd.DataFrame(columns=["reservoir_id", "polygon_area_km2",
                                              "polygon_area_source"]))
    return out, sorted(missing_packages)


def attach_polygons(table, reaches, areas):
    """table (AR_ID, Year_Removed) plus reservoir_status, reservoir_id,
    polygon_timing, polygon_area_km2, polygon_area_source and
    polygon_area_km2_pre_removal. Areas and timing only for dams at a
    reservoir outlet; everything else null."""
    from etl.common.water_features import clean_permanent_id

    out = table.merge(reaches.drop(columns=["huc8"], errors="ignore"), on="AR_ID", how="left")
    at_outlet = out["reservoir_status"].eq(AT_OUTLET).fillna(False).astype(bool)
    out["polygon_timing"] = polygon_timing(out["reservoir_fdate"], out["Year_Removed"]) \
        .where(at_outlet)
    if areas is not None and len(areas):
        areas = areas.assign(_rid=clean_permanent_id(areas["reservoir_id"])) \
            .drop(columns="reservoir_id").dropna(subset=["_rid"]).drop_duplicates("_rid")
        out["_rid"] = clean_permanent_id(out["reservoir_id"])
        out = out.merge(areas, on="_rid", how="left").drop(columns="_rid")
    else:
        out["polygon_area_km2"] = np.nan
        out["polygon_area_source"] = pd.NA
    out["polygon_area_km2"] = pd.to_numeric(out["polygon_area_km2"], errors="coerce") \
        .where(at_outlet)
    out["polygon_area_source"] = out["polygon_area_source"].where(at_outlet)
    out["polygon_area_km2_pre_removal"] = out["polygon_area_km2"].where(
        out["polygon_timing"].eq(PRE_REMOVAL).fillna(False).astype(bool))
    return out
