"""
Build dam_reservoir.parquet: the mapped reservoir polygon of each dam in
dam_removal (docs/warehouse_spec.md, section 3.12). Added 2026-11-04 so
analysis_measurement can carry measured reservoir area as a supplementary
size variable next to the constructed ones (backwater_km,
reservoir_volume_proxy in build_views.link_features).

Reads Stage 4 outputs (data/processed/dam_reaches/<case_study>/, from
match_dams.py) and, for dams at a reservoir outlet, the NHDWaterbody
polygons in each HUC8's extracted NHDPlus HR package. No reach matching is
redone and no ETL stage before it changes. Rules: etl/common/reservoir.py.

Columns (one row per dam_removal AR_ID, also dams without a polygon):
    AR_ID
    reservoir_status        Stage 4: dam_at_outlet / reservoir_not_at_outlet /
                            no_reservoir_polygon; null when the dam isn't in
                            the Stage 4 outputs (e.g. link_source: snapped)
    reservoir_id            NHDWaterbody Permanent_Identifier
    polygon_timing          edited_by_removal_year / edited_after_removal /
                            date_unknown (polygon's NHD FDate vs removal year)
    polygon_area_km2        measured polygon area; null unless dam_at_outlet
    polygon_area_source     AreaSqKm or geometry_5070
Nothing is imputed: a dam without a mapped polygon keeps null values.

Run after build_views.py (it adds its view to warehouse.duckdb) and before
build_analysis_dataset.py, which requires it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.build_views import add_view
from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.reach_inputs import dam_id_text
from etl.common.reservoir import AT_OUTLET, attach_polygons, read_dam_reaches, reservoir_areas
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic

SCHEMA = {
    "AR_ID": "string",
    "reservoir_status": "string",
    "reservoir_id": "string",
    "polygon_timing": "string",
    "polygon_area_km2": "float64",
    "polygon_area_source": "string",
}
PRIMARY_KEY = ["AR_ID"]


def build_table(dams, reaches, areas):
    """One row per dam (dams: AR_ID, Year_Removed) with SCHEMA's columns."""
    dams = dams.assign(AR_ID=dam_id_text(dams["AR_ID"]))
    out = attach_polygons(dams[["AR_ID", "Year_Removed"]], reaches, areas)
    return out[list(SCHEMA)].astype(SCHEMA)


def validate(table, dams):
    errors = schema_errors("dam_reservoir", table, SCHEMA, PRIMARY_KEY)
    if errors:
        return errors
    if set(table["AR_ID"]) != set(dam_id_text(dams["AR_ID"])):
        errors.append("AR_IDs differ from dam_removal")
    if table.loc[table["reservoir_status"].ne(AT_OUTLET).fillna(True),
                 "polygon_area_km2"].notna().any():
        errors.append("polygon area set for a dam not at a reservoir outlet")
    if (table["polygon_area_km2"] <= 0).any():
        errors.append("polygon_area_km2 <= 0")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study(),
                        help="Name of a config file under config/ (without .yaml).")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    output_dir = warehouse_dir(cfg)
    db_path = require_file(output_dir / "warehouse.duckdb",
                           "warehouse.duckdb (run build_views.py first)")
    dams = pd.read_parquet(require_file(output_dir / "dam_removal.parquet",
                                        "dam_removal table (run load_core_tables.py first)"),
                           columns=["AR_ID", "Year_Removed"])
    processed = cfg.project_root / "data" / "processed"
    reaches = read_dam_reaches(processed / "dam_reaches" / cfg.name)
    at_outlet = reaches[reaches["reservoir_status"].eq(AT_OUTLET)]
    areas, missing = reservoir_areas(at_outlet, processed / "nhdplus_extracted") \
        if len(at_outlet) else (None, [])

    table = build_table(dams, reaches, areas)
    if errors := validate(table, dams):
        raise ValueError("Validation failed; no warehouse files were written:\n  - "
                         + "\n  - ".join(errors))
    path = output_dir / "dam_reservoir.parquet"
    write_parquet_atomic(table, path)
    add_view(db_path, "dam_reservoir", path)

    print(f"Case study: {cfg.name}")
    print(f"Wrote {len(table):,} dams to {path}")
    print(f"Stage 4 dam_reaches rows: {len(reaches):,}; dams in dam_removal not in them: "
          f"{int(table['reservoir_status'].isna().sum()):,}")
    if missing:
        print(f"HUC8s with a reservoir but no NHD package found: {missing}")
    print("\nreservoir_status:")
    print(table["reservoir_status"].value_counts(dropna=False).to_string())
    print("\npolygon_timing (dams at an outlet):")
    print(table["polygon_timing"].value_counts(dropna=False).to_string())
    area = table["polygon_area_km2"]
    print(f"\nWith a polygon area: {int(area.notna().sum()):,} of {len(table):,} dams; "
          f"mapped by the removal year: "
          f"{int((area.notna() & table['polygon_timing'].eq('edited_by_removal_year')).sum()):,}")
    print(area.describe(percentiles=[0.1, 0.5, 0.9]).to_string())


if __name__ == "__main__":
    main()
