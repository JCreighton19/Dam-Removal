"""
Build station_watershed.parquet: the upstream drainage area of every
station that enters the analysis (docs/warehouse_spec.md, section 3.7).

Which stations
--------------
Stations with `has_both` for at least one parameter in the
eligibility_pairs view of warehouse.duckdb (build_views.py), i.e. the
downstream sites and upstream controls the analysis will use. Land cover
and climate (later stages) are computed only for these stations.

How the upstream area is built
------------------------------
1. Read every case-study NHDPlus HR geodatabase and stitch them into one
   reach graph (FromNHDPID -> ToNHDPID is downstream, as everywhere else in
   the pipeline). NHDPlusIDs are national, so reaches crossing between
   packages connect.
2. A station's upstream reaches are its own reach plus every graph
   ancestor. Each reach has one NHDPlusCatchment polygon (the land that
   drains to it), with the same NHDPlusID.
3. area_km2 is the sum of those catchments' published AreaSqKm.
4. The station -> catchment membership is also written to
   station_catchment.parquet, so the land-cover and climate stages reuse it
   instead of rebuilding the reach graph.
5. With --with-outlines, geometry is the dissolved union of the catchments
   (for map QA). Merging tens of thousands of catchments per large river is
   slow, so it is off by default and geometry is null.

The station's own catchment is included whole, even though the station
may sit partway along its reach.

Incomplete upstream areas
-------------------------
A large river's upstream area can extend past the downloaded NHDPlus
packages. That is flagged two ways:
- nhdplus_total_area_km2 is NHDPlus's own published total drainage area
  for the station's reach (NHDPlusFlowlineVAA.TotDASqKm), and
  area_coverage = area_km2 / nhdplus_total_area_km2.
- truncated is True when a flow row enters an upstream reach from a reach
  that isn't in any downloaded package, or when area_coverage < 0.9.

Invalid areas
-------------
NHDPlus uses -9999 (and other non-positive values) as "no value". They are
never stored or summed as areas:
- a catchment AreaSqKm placeholder is left out of area_km2, and the station
  is flagged invalid_catchment_area;
- a TotDASqKm placeholder (or no VAA layer) leaves nhdplus_total_area_km2
  null, so completeness can't be checked: nhdplus_area_missing.
drainage_area_flag is 'ok' only when neither applies and the area isn't
truncated; analysis_measurement uses area_km2 only then.

catchment_hash identifies each station's set of upstream catchments; the
land-cover and climate stages keep rows of stations whose hash is unchanged
(etl/common/covariates.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import geopandas as gpd
import networkx as nx
import pandas as pd
import pyogrio
from shapely import make_valid, union_all

from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.network import build_reach_graph
from etl.common.nhdplus import find_huc8_geodatabase, read_nhdplus_layer
from etl.common.progress import huc8s_with_outputs
from etl.build_views import add_view
from etl.common.covariates import catchment_hashes, keep_previous_membership
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic

WGS84 = "EPSG:4326"

# Below this share of NHDPlus's published drainage area, the upstream area
# is treated as incomplete.
MIN_AREA_COVERAGE = 0.9

# Dissolved polygons are simplified (in meters) to keep the file small;
# they are for maps only.
SIMPLIFY_M = 30

CATCHMENT_SCHEMA = {
    "station_id": "string",
    "catchment_id": "int64",
}

SCHEMA = {
    "station_id": "string",
    "NHDPlusID": "int64",
    "area_km2": "float64",
    "n_catchments": "int64",
    "nhdplus_total_area_km2": "float64",
    "area_coverage": "float64",
    "truncated": "bool",
    "drainage_area_flag": "string",
    "catchment_hash": "string",
    "geometry": "geometry",
}

# Worst first: the flag a station gets when several apply.
DRAINAGE_AREA_FLAGS = ["invalid_catchment_area", "truncated", "nhdplus_area_missing", "ok"]


# --------------------------------------------------
# Inputs
# --------------------------------------------------

def eligible_stations(db_path):
    """Stations with has_both for any parameter, with their snapped reach."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.sql("""
            SELECT station_id, s.NHDPlusID, s.huc8
            FROM wq_station s
            WHERE station_id IN (SELECT station_id FROM eligibility_pairs WHERE has_both)
            ORDER BY station_id
        """).df()


def _lower(frame):
    return frame.rename(columns=str.lower)


def _read_table(gdb_path, layer, columns, **kwargs):
    """Read columns whatever their case in this geodatabase, lowercased."""
    fields = {str(f).lower(): str(f) for f in pyogrio.read_info(gdb_path, layer=layer)["fields"]}
    missing = [c for c in columns if c.lower() not in fields]
    if missing:
        raise ValueError(f"Layer '{layer}' in {gdb_path} is missing columns {missing}")
    frame = pyogrio.read_dataframe(
        gdb_path, layer=layer, columns=[fields[c.lower()] for c in columns],
        read_geometry=False, **kwargs,
    )
    return _lower(frame)


def _ids(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def read_network(gdb_paths):
    """Reach IDs/lengths, flow rows, catchment areas and VAA drainage areas
    from every geodatabase, combined and de-duplicated."""
    reaches, flows, catchments, vaas = [], [], [], []

    for gdb_path in gdb_paths:
        print(f"Reading {gdb_path.name}", flush=True)
        reaches.append(read_nhdplus_layer(
            gdb_path, "NHDFlowline", columns=["NHDPlusID", "LengthKM"], ignore_geometry=True
        ))
        flows.append(read_nhdplus_layer(
            gdb_path, "NHDPlusFlow", columns=["FromNHDPID", "ToNHDPID"], ignore_geometry=True
        ))

        catchment = _read_table(
            gdb_path, "NHDPlusCatchment", ["NHDPlusID", "AreaSqKm"], fid_as_index=True
        )
        catchment = catchment.rename_axis("fid").reset_index()
        catchment["gdb_path"] = str(gdb_path)
        catchments.append(catchment)

        layers = {name.lower() for name, _ in pyogrio.list_layers(gdb_path)}
        if "nhdplusflowlinevaa" in layers:
            vaas.append(_read_table(gdb_path, "NHDPlusFlowlineVAA", ["NHDPlusID", "TotDASqKm"]))

    reaches = pd.concat(reaches, ignore_index=True)
    reaches["NHDPlusID"] = _ids(reaches["NHDPlusID"])
    reaches = reaches.dropna(subset=["NHDPlusID"]).drop_duplicates("NHDPlusID")

    flow = pd.concat(flows, ignore_index=True)
    flow["FromNHDPID"] = _ids(flow["FromNHDPID"])
    flow["ToNHDPID"] = _ids(flow["ToNHDPID"])
    flow = flow.drop_duplicates()

    catchments = pd.concat(catchments, ignore_index=True)
    catchments["nhdplusid"] = _ids(catchments["nhdplusid"])
    catchments = catchments.dropna(subset=["nhdplusid"]).drop_duplicates("nhdplusid")
    catchments = catchments.set_index("nhdplusid")

    vaa = pd.concat(vaas, ignore_index=True) if vaas else pd.DataFrame(
        {"nhdplusid": [], "totdasqkm": []}
    )
    vaa["nhdplusid"] = _ids(vaa["nhdplusid"])
    total_area = (
        vaa.dropna(subset=["nhdplusid"]).drop_duplicates("nhdplusid")
        .set_index("nhdplusid")["totdasqkm"].astype("float64")
    )
    # -9999 and other non-positive values are NHDPlus placeholders.
    total_area = total_area.where(total_area > 0)

    return reaches, flow, catchments, total_area


def read_catchment_geometries(catchments, ids):
    """Polygons for the given catchment IDs, reading only those features."""
    wanted = catchments.loc[catchments.index.intersection(list(ids))]
    frames = []
    for gdb_path, rows in wanted.groupby("gdb_path"):
        frame = pyogrio.read_dataframe(gdb_path, layer="NHDPlusCatchment",
                                       fids=rows["fid"].tolist(), columns=[])
        frame.index = rows.index
        frames.append(frame.to_crs(WGS84))
    if not frames:
        return gpd.GeoSeries([], crs=WGS84)
    return gpd.GeoDataFrame(pd.concat(frames), crs=WGS84).geometry


# --------------------------------------------------
# Upstream areas
# --------------------------------------------------

def upstream_reaches(graph, reach):
    """The reach itself plus every reach that drains into it."""
    if reach not in graph:
        return {reach}
    return nx.ancestors(graph, reach) | {reach}


def reaches_with_outside_inflow(flow, known_reaches):
    """Reaches that water enters from a reach in no downloaded package."""
    source = flow["FromNHDPID"]
    outside = source.notna() & source.ne(0) & ~source.isin(known_reaches)
    return set(flow.loc[outside, "ToNHDPID"].dropna().astype("int64"))


def dissolve(polygons):
    if polygons.empty:
        return None
    merged = gpd.GeoSeries([make_valid(union_all(polygons.values))], crs=WGS84)
    metric_crs = merged.estimate_utm_crs()
    return merged.to_crs(metric_crs).simplify(SIMPLIFY_M).to_crs(WGS84).iloc[0]


def build_station_watershed(stations, reaches, flow, catchments, total_area,
                            with_outlines=False):
    """Returns (station_watershed table, station_catchment table)."""
    graph, _, stats = build_reach_graph(reaches, flow)
    print(f"Reach graph: {stats['stream_count']:,} reaches, "
          f"{graph.number_of_edges():,} connections", flush=True)

    known_reaches = set(reaches["NHDPlusID"].astype("int64"))
    outside_entry = reaches_with_outside_inflow(flow, known_reaches)
    upstream = {
        row.station_id: upstream_reaches(graph, int(row.NHDPlusID))
        for row in stations.itertuples(index=False)
    }

    if with_outlines:
        all_ids = set().union(*upstream.values()) if upstream else set()
        geometries = read_catchment_geometries(catchments, all_ids)
    area_by_catchment = pd.to_numeric(catchments["areasqkm"], errors="coerce").astype("float64")
    valid_catchment = area_by_catchment > 0  # -9999 etc. are placeholders

    rows, memberships = [], []
    for row in stations.itertuples(index=False):
        ids = upstream[row.station_id]
        in_catchments = catchments.index.intersection(list(ids))
        valid = valid_catchment.loc[in_catchments]
        area = float(area_by_catchment.loc[in_catchments][valid].sum())
        nhdplus_total = total_area.get(int(row.NHDPlusID))
        nhdplus_total = None if nhdplus_total is None or pd.isna(nhdplus_total) else nhdplus_total
        coverage = area / nhdplus_total if nhdplus_total else None
        truncated = not outside_entry.isdisjoint(ids) or (
            coverage is not None and coverage < MIN_AREA_COVERAGE
        )
        flag = ("invalid_catchment_area" if not valid.all()
                else "truncated" if truncated
                else "nhdplus_area_missing" if nhdplus_total is None
                else "ok")
        rows.append({
            "station_id": row.station_id,
            "NHDPlusID": int(row.NHDPlusID),
            "area_km2": area,
            "n_catchments": len(in_catchments),
            "nhdplus_total_area_km2": nhdplus_total,
            "area_coverage": coverage,
            "truncated": truncated,
            "drainage_area_flag": flag,
            "catchment_hash": None,
            "geometry": (
                dissolve(geometries.loc[geometries.index.intersection(in_catchments)])
                if with_outlines else None
            ),
        })
        memberships.append(pd.DataFrame({
            "station_id": row.station_id,
            "catchment_id": in_catchments.astype("int64"),
        }))
        print(f"  {row.station_id}: {area:,.1f} km2 from {len(in_catchments):,} catchments"
              + ("" if flag == "ok" else f" ({flag})"), flush=True)

    table = gpd.GeoDataFrame(rows, columns=list(SCHEMA), geometry="geometry", crs=WGS84)
    for column, dtype in SCHEMA.items():
        if column != "geometry":
            table[column] = table[column].astype(dtype)

    membership = (
        pd.concat(memberships, ignore_index=True) if memberships
        else pd.DataFrame(columns=list(CATCHMENT_SCHEMA))
    ).astype(CATCHMENT_SCHEMA)
    table["catchment_hash"] = table["station_id"].map(catchment_hashes(membership)) \
        .astype("string")
    return table, membership


# --------------------------------------------------
# Validation (spec section 4)
# --------------------------------------------------

def validate(table, membership, station_ids):
    errors = schema_errors("station_watershed", table, SCHEMA, ["station_id"])
    errors += schema_errors("station_catchment", membership, CATCHMENT_SCHEMA,
                            ["station_id", "catchment_id"])
    if errors:
        return errors
    if bad := sorted(set(table["station_id"]) - set(station_ids))[:10]:
        errors.append(f"station_id not in wq_station, e.g. {bad}")
    if not (table["area_km2"] > 0).all():
        bad = table.loc[~(table["area_km2"] > 0), "station_id"].head(10).tolist()
        errors.append(f"area_km2 not > 0 (no catchments found), e.g. {bad}")
    if (table["nhdplus_total_area_km2"].dropna() <= 0).any():
        errors.append("nhdplus_total_area_km2 has a non-positive (placeholder) value")
    if not table["drainage_area_flag"].isin(DRAINAGE_AREA_FLAGS).all():
        errors.append("drainage_area_flag has unexpected values")
    if table["catchment_hash"].isna().any():
        errors.append("catchment_hash is null")
    return errors


# --------------------------------------------------
# Main
# --------------------------------------------------

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
        "--with-outlines",
        action="store_true",
        help="Also dissolve each station's catchments into an outline polygon "
        "(for map QA; slow for large rivers).",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    output_dir = warehouse_dir(cfg)
    db_path = require_file(
        output_dir / "warehouse.duckdb", "warehouse.duckdb (run build_views.py first)"
    )

    print(f"Case study: {cfg.name}")
    stations = eligible_stations(db_path)
    print(f"Eligible stations (has_both for any parameter): {len(stations):,}")

    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_paths = sorted({find_huc8_geodatabase(nhd_dir, huc8) for huc8 in huc8s_with_outputs(cfg)})
    reaches, flow, catchments, total_area = read_network(gdb_paths)

    table, membership = build_station_watershed(
        stations, reaches, flow, catchments, total_area, with_outlines=args.with_outlines
    )

    with duckdb.connect(str(db_path), read_only=True) as con:
        station_ids = con.sql("SELECT station_id FROM wq_station").df()["station_id"]
    errors = validate(table, membership, station_ids)
    if errors:
        raise ValueError(
            "Validation failed; no warehouse files were written:\n  - "
            + "\n  - ".join(errors)
        )

    path = output_dir / "station_watershed.parquet"
    write_parquet_atomic(table, path)
    add_view(db_path, "station_watershed", path)
    print(f"\nWrote {len(table):,} rows to {path}")
    print("Drainage area flags:")
    print(table["drainage_area_flag"].value_counts().to_string())

    membership_path = output_dir / "station_catchment.parquet"
    if keep_previous_membership(output_dir):
        print("Kept the previous station_catchment as station_catchment.prev.parquet, so "
              "existing land_cover/climate rows can be reused")
    write_parquet_atomic(membership, membership_path)
    add_view(db_path, "station_catchment", membership_path)
    print(f"Wrote {len(membership):,} station-catchment rows to {membership_path}")


if __name__ == "__main__":
    main()
