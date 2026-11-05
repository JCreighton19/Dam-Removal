"""
Stage 1A: reservoir-size audit (read-only; writes only to
data/processed/stage1_audit/<case_study>/).

What it measures, per dam removal
---------------------------------
Constructed from the warehouse (no new data):
    height_m                Dam_Height_ft x 0.3048 (null if missing or <= 0)
    backwater_km            height_m / reach_slope / 1000 -- the same formula
                            and slope rule as build_views.link_features: the
                            length of a flat pool behind a dam of that height
                            on a channel of that slope. Null if the slope is
                            missing or at or below the NHDPlus minimum
                            (build_views.MIN_BACKWATER_SLOPE, 1e-5).
    reservoir_volume_proxy  height_m^2 / reach_slope * sqrt(reach_totdasqkm),
                            in m^2 * km^0.5. An index, not a volume: a wedge
                            of height H and length H/S has a cross-section of
                            about H^2/S, and channel width grows roughly with
                            the square root of drainage area. Only its ranks /
                            logs mean anything.
    slope_at_floor          reach_slope <= 1e-5. NHDPlus gives flat or
                            uncomputable reaches that minimum slope, which
                            made backwater_km and the proxy implausibly large
                            (up to 3,292 km; first national run, 2026-11-04).
                            Since 2026-11-04 both are null for these dams, as
                            in build_views; the flag says how many.

Measured (kept separate from the constructed ones; never imputed):
    reservoir polygon       Stage 4 (match_dams.py): the NHDWaterbody whose
                            main outlet the dam sits at, read as in
                            etl/common/reservoir.py (shared with
                            build_dam_reservoir.py): area, and polygon_timing
                            edited_by_removal_year / edited_after_removal /
                            date_unknown from the polygon's NHD FDate.
    NID / NABD storage      optional --storage-file (CSV, or any file
                            pyogrio reads) with one row per dam: an NID id
                            column and storage / surface-area columns. Matched
                            on the dam list's NID_ID only (load_core_tables
                            doesn't carry it, so it is read from the dam CSV).
                            NID and NABD are both keyed by the NID id; NABI_ID
                            is a different inventory's id and is never used
                            for the join (before 2026-11-04 it was tried as a
                            fallback). Columns are detected from common NID /
                            NABD names or set with --id-column /
                            --storage-column / --area-column.

Outputs
-------
    reservoir_size_by_dam.csv       one row per dam in the dam list
    reservoir_coverage.csv          n and % of dams with each variable, for
                                    all removals and for dams in dam_removal
    reservoir_correlations.csv      log10 Pearson and Spearman between each
                                    measured variable and each constructed one
                                    (n < --min-n: reported as insufficient)
    reservoir_selection_bias.csv    dams with vs without each measured
                                    variable: medians of height, drainage
                                    area, slope, removal year; share analyzed
    reservoir_collinearity.csv      dams in dam_removal: log10 correlation
                                    and VIF of each size measure with log10
                                    dam drainage area (the size model's
                                    covariate; the proxy contains it)

Run on the cluster through data/slurm/stage1_audits.slurm (AUDIT=reservoir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis.audit_utils import save_output
from etl.build_views import MIN_BACKWATER_SLOPE
from etl.common.config import default_case_study, load_config
from etl.common.reach_inputs import dam_id_text
from etl.common.reservoir import attach_polygons, polygon_timing, read_dam_reaches  # noqa: F401
from etl.common.reservoir import reservoir_areas
from etl.common.warehouse import warehouse_dir

FT_TO_M = 0.3048
ACRE_TO_KM2 = 0.00404686
SLOPE_FLOOR = MIN_BACKWATER_SLOPE

# Common column names in NID (2022+ national CSV) and NABD (Ostroff et al.
# 2013) exports, matched ignoring case. Override with the CLI flags.
ID_CANDIDATES = ["NIDID", "NID_ID", "NID ID", "Federal ID", "FEDERAL_ID", "FEDERALID"]
STORAGE_CANDIDATES = {
    "normal_storage_acre_ft": ["NORMAL_STORAGE", "Normal Storage (Acre-Ft)", "NORM_STOR",
                               "NORMSTOR", "NORMAL_STOR"],
    "nid_storage_acre_ft": ["NID_STORAGE", "NID Storage (Acre-Ft)", "NID_STORAG", "NIDSTORAGE"],
    "max_storage_acre_ft": ["MAX_STORAGE", "Max Storage (Acre-Ft)", "MAX_STOR", "MAXSTOR"],
}
AREA_CANDIDATES = ["SURFACE_AREA", "Surface Area (Acres)", "SURF_AREA", "SURFACEAREA"]

CONSTRUCTED = ["height_m", "backwater_km", "reservoir_volume_proxy"]


# --------------------------------------------------
# Constructed size variables
# --------------------------------------------------

def size_variables(frame):
    """height_m, backwater_km, reservoir_volume_proxy and slope_at_floor from
    Dam_Height_ft, reach_slope and reach_totdasqkm (non-positive -> null;
    slopes at or below SLOPE_FLOOR -> null backwater and proxy, as in
    build_views.link_features)."""
    height = pd.to_numeric(frame["Dam_Height_ft"], errors="coerce") * FT_TO_M
    height = height.where(height > 0)
    slope = pd.to_numeric(frame["reach_slope"], errors="coerce")
    slope = slope.where(slope > 0)
    usable_slope = slope.where(slope > SLOPE_FLOOR)
    area = pd.to_numeric(frame["reach_totdasqkm"], errors="coerce")
    area = area.where(area > 0)
    return frame.assign(
        height_m=height,
        backwater_km=height / usable_slope / 1000,
        reservoir_volume_proxy=height ** 2 / usable_slope * np.sqrt(area),
        slope_at_floor=slope.le(SLOPE_FLOOR).astype("boolean").where(slope.notna()),
    )


# --------------------------------------------------
# NID / NABD storage file
# --------------------------------------------------

def find_column(columns, candidates, override=None):
    """The column named `override` (must exist), else the first candidate
    present (ignoring case and surrounding spaces), else None."""
    by_lower = {str(c).strip().lower(): c for c in columns}
    if override:
        if override.strip().lower() not in by_lower:
            raise ValueError(f"Column {override!r} not in the storage file. "
                             f"Columns: {list(columns)}")
        return by_lower[override.strip().lower()]
    return next((by_lower[c.lower()] for c in candidates if c.lower() in by_lower), None)


def normalize_id(series):
    text = series.astype("string").str.strip().str.upper()
    return text.mask(text.isin(["", "NAN", "NONE"]))


def read_storage_file(path, id_column=None, storage_column=None, area_column=None):
    """One row per id: storage_id plus whichever storage / area columns exist
    (acre-ft; surface area converted to km2). Returns (frame, notes)."""
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(path, low_memory=False)
    else:
        import pyogrio
        raw = pyogrio.read_dataframe(path, read_geometry=False)
    notes = [f"storage file: {path} ({len(raw):,} rows)"]
    id_col = find_column(raw.columns, ID_CANDIDATES, id_column)
    if id_col is None:
        raise ValueError(f"No NID id column found in {path}; pass --id-column. "
                         f"Columns: {list(raw.columns)}")
    out = pd.DataFrame({"storage_id": normalize_id(raw[id_col])})
    notes.append(f"id column: {id_col!r}")
    for name, candidates in STORAGE_CANDIDATES.items():
        override = storage_column if (storage_column and name == "normal_storage_acre_ft") else None
        column = find_column(raw.columns, candidates, override)
        if column is not None:
            values = pd.to_numeric(raw[column], errors="coerce")
            out[name] = values.where(values > 0)
            notes.append(f"{name}: {column!r}")
    area_col = find_column(raw.columns, AREA_CANDIDATES, area_column)
    if area_col is not None:
        acres = pd.to_numeric(raw[area_col], errors="coerce")
        out["nid_surface_area_km2"] = acres.where(acres > 0) * ACRE_TO_KM2
        notes.append(f"nid_surface_area_km2: {area_col!r} (acres x {ACRE_TO_KM2})")
    if len(out.columns) == 1:
        raise ValueError(f"No storage or surface-area column found in {path}; pass "
                         f"--storage-column / --area-column. Columns: {list(raw.columns)}")
    out = out.dropna(subset=["storage_id"])
    duplicated = int(out["storage_id"].duplicated().sum())
    notes.append(f"duplicate ids (first kept): {duplicated:,}")
    return out.drop_duplicates("storage_id"), notes


def measured_storage(row_ids, storage):
    """Storage columns for each dam id (aligned with row_ids; null if not
    matched), and a matched flag."""
    found = storage.set_index("storage_id").reindex(normalize_id(row_ids).to_numpy())
    found.index = row_ids.index
    return found, normalize_id(row_ids).isin(set(storage["storage_id"]))


# --------------------------------------------------
# Stage 4 reservoir polygons
# --------------------------------------------------

# read_dam_reaches, reservoir_areas, attach_polygons: etl/common/reservoir.py


# --------------------------------------------------
# Summaries
# --------------------------------------------------

def has_value(values):
    """True / false flags as themselves (null = false); anything else: non-null."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.notna()

def coverage(table, variables, universes):
    """n and % of dams with a non-null (or true) value, per universe."""
    rows = []
    for universe, mask in universes.items():
        subset = table[mask]
        for variable in variables:
            has = has_value(subset[variable])
            rows.append({"universe": universe, "variable": variable, "dams": len(subset),
                         "with_value": int(has.sum()),
                         "pct": round(100 * has.mean(), 1) if len(subset) else np.nan})
    return pd.DataFrame(rows)


def correlations(table, measured, constructed, min_n=10):
    """log10 Pearson and Spearman between each measured and constructed
    variable over dams with both (positive) values."""
    rows = []
    for m in measured:
        for c in constructed:
            if m not in table or c not in table:
                continue
            pair = table[[m, c]].apply(pd.to_numeric, errors="coerce")
            pair = pair[(pair > 0).all(axis=1)]
            row = {"measured": m, "constructed": c, "n": len(pair)}
            if len(pair) >= min_n:
                logs = np.log10(pair)
                row.update(pearson_log10=round(logs[m].corr(logs[c]), 3),
                           spearman=round(pair[m].corr(pair[c], method="spearman"), 3),
                           note="")
            else:
                row.update(pearson_log10=np.nan, spearman=np.nan,
                           note=f"insufficient (n < {min_n})")
            rows.append(row)
    return pd.DataFrame(rows)


def selection_bias(table, groups, describe):
    """Dams with vs without each measured variable: n, medians of `describe`
    and the share in the analysis (dam_removal)."""
    rows = []
    for group in groups:
        if group not in table:
            continue
        has = has_value(table[group])
        for label, mask in (("with", has), ("without", ~has)):
            subset = table[mask]
            row = {"variable": group, "group": label, "dams": len(subset),
                   "pct_in_analysis": round(100 * subset["in_analysis"].mean(), 1)
                   if len(subset) else np.nan}
            for column in describe:
                values = pd.to_numeric(subset[column], errors="coerce")
                row[f"median_{column}"] = values.median()
                row[f"pct_missing_{column}"] = round(100 * values.isna().mean(), 1) \
                    if len(subset) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def collinearity(table, sizes, covariate="reach_totdasqkm", min_n=10):
    """log10 Pearson r and VIF (1 / (1 - r^2), two predictors) between each
    size measure and the covariate, over dams with both positive."""
    rows = []
    for size in sizes:
        if size not in table:
            continue
        pair = table[[size, covariate]].apply(pd.to_numeric, errors="coerce")
        pair = pair[(pair > 0).all(axis=1)]
        row = {"size": size, "covariate": f"log10 {covariate}", "n": len(pair)}
        if len(pair) >= min_n:
            r = np.log10(pair[size]).corr(np.log10(pair[covariate]))
            row.update(pearson_log10=round(r, 3),
                       vif=round(1 / (1 - r ** 2), 2) if abs(r) < 1 else np.inf, note="")
        else:
            row.update(pearson_log10=np.nan, vif=np.nan, note=f"insufficient (n < {min_n})")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------
# Main
# --------------------------------------------------

def build(cfg, storage_file=None, id_column=None, storage_column=None, area_column=None,
          read_polygons=True):
    """The per-dam audit table and notes."""
    notes = []
    dam_list = pd.read_csv(cfg.path("dam_source"), low_memory=False, dtype=str)
    dam_list["AR_ID"] = dam_id_text(dam_list["AR_ID"])
    dam_list = dam_list.drop_duplicates("AR_ID")
    keep = [c for c in ["AR_ID", "NABI_ID", "NID_ID", "State", "Year_Removed", "Dam_Height_ft"]
            if c in dam_list]
    table = dam_list[keep].copy()
    table["Year_Removed"] = pd.to_numeric(table.get("Year_Removed"), errors="coerce")
    table = table[table["Year_Removed"].notna()].copy()
    notes.append(f"dam list: {len(dam_list):,} rows, {len(table):,} with a removal year")

    warehouse = pd.read_parquet(warehouse_dir(cfg) / "dam_removal.parquet",
                                columns=["AR_ID", "Dam_Height_ft", "reach_slope",
                                         "reach_totdasqkm"])
    warehouse["AR_ID"] = dam_id_text(warehouse["AR_ID"])
    table = table.drop(columns="Dam_Height_ft", errors="ignore").merge(
        warehouse, on="AR_ID", how="left", indicator=True)
    table["in_analysis"] = table.pop("_merge").eq("both")
    # Height for dams outside dam_removal comes from the dam list itself.
    listed_height = dam_list.set_index("AR_ID")["Dam_Height_ft"] if "Dam_Height_ft" in dam_list \
        else pd.Series(dtype=str)
    table["Dam_Height_ft"] = table["Dam_Height_ft"].fillna(
        pd.to_numeric(table["AR_ID"].map(listed_height), errors="coerce"))
    table = size_variables(table)

    reaches = read_dam_reaches(cfg.project_root / "data" / "processed" / "dam_reaches" / cfg.name)
    notes.append(f"Stage 4 dam_reaches: {len(reaches):,} dams")
    areas = None
    if read_polygons and len(reaches):
        areas, missing = reservoir_areas(reaches[reaches["reservoir_status"].eq("dam_at_outlet")],
                                         cfg.project_root / "data" / "processed"
                                         / "nhdplus_extracted")
        if missing:
            notes.append(f"HUC8s with a reservoir but no NHD package found: {missing}")
    table = attach_polygons(table, reaches, areas)

    if storage_file:
        storage, storage_notes = read_storage_file(storage_file, id_column, storage_column,
                                                   area_column)
        notes += storage_notes
        values = [c for c in storage.columns if c != "storage_id"]
        # NID_ID only: NID and NABD are keyed by the NID id; NABI_ID is not one.
        if "NID_ID" not in table:
            raise ValueError("The dam list has no NID_ID column; storage can't be matched.")
        found, matched = measured_storage(table["NID_ID"], storage)
        table["storage_matched_by_NID_ID"] = matched
        notes.append(f"matched on NID_ID: {int(matched.sum()):,} of "
                     f"{int(table['NID_ID'].notna().sum()):,} dams with a NID_ID")
        for column in values:
            table[column] = found[column]
    return table, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--storage-file", type=Path, help="NID / NABD export (optional).")
    parser.add_argument("--id-column")
    parser.add_argument("--storage-column", help="Normal-storage column, if not detected.")
    parser.add_argument("--area-column", help="Surface-area column (acres), if not detected.")
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--tag", default="",
                        help="Suffix for the output names, e.g. _nabd, so runs with different "
                             "storage files don't overwrite each other.")
    parser.add_argument("--skip-polygons", action="store_true",
                        help="Don't read NHD packages for polygon areas.")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    out_dir = cfg.project_root / "data" / "processed" / "stage1_audit" / cfg.name
    table, notes = build(cfg, args.storage_file, args.id_column, args.storage_column,
                         args.area_column, read_polygons=not args.skip_polygons)

    measured = [c for c in ["polygon_area_km2_pre_removal", "polygon_area_km2",
                            "normal_storage_acre_ft", "nid_storage_acre_ft",
                            "max_storage_acre_ft", "nid_surface_area_km2"] if c in table]
    flags = [c for c in ["storage_matched_by_NID_ID"] if c in table]
    variables = (["NID_ID", "NABI_ID"] + flags + ["Dam_Height_ft", "reach_slope",
                 "reach_totdasqkm"] + CONSTRUCTED + ["slope_at_floor", "reservoir_id"]
                 + measured)
    variables = [v for v in variables if v in table]
    universes = {"all removals (dam list, removal year)": pd.Series(True, index=table.index),
                 "dams in dam_removal (analysis)": table["in_analysis"]}
    cov = coverage(table, variables, universes)
    cor = correlations(table[table["in_analysis"]], measured, CONSTRUCTED, args.min_n)
    bias = selection_bias(table, ["reservoir_id", "polygon_area_km2_pre_removal"] + flags
                          + [m for m in measured if "storage" in m],
                          ["height_m", "reach_totdasqkm", "reach_slope", "Year_Removed",
                           "reservoir_volume_proxy"])

    collinear = collinearity(table[table["in_analysis"]],
                             ["reservoir_volume_proxy", "height_m", "backwater_km"] + measured,
                             min_n=args.min_n)

    names = [f"{stem}{args.tag}.csv" for stem in ("reservoir_size_by_dam", "reservoir_coverage",
                                                  "reservoir_correlations",
                                                  "reservoir_selection_bias",
                                                  "reservoir_collinearity")]
    for name, frame in zip(names, [table, cov, cor, bias, collinear]):
        save_output(frame, name, out_dir)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Case study: {cfg.name}\n" + "\n".join(f"  {n}" for n in notes))
    print("\n=== Coverage ===")
    print(cov.to_string(index=False))
    print("\n=== Stage 4 reservoir polygons (dams in dam_removal) ===")
    analyzed = table[table["in_analysis"]]
    print(analyzed["reservoir_status"].value_counts(dropna=False).to_string())
    print(analyzed["polygon_timing"].value_counts(dropna=False).to_string())
    if "polygon_area_source" in analyzed:
        print(analyzed["polygon_area_source"].value_counts(dropna=False).to_string())
    print("\n=== Constructed variables (dams in dam_removal) ===")
    print(analyzed[CONSTRUCTED].describe(percentiles=[0.1, 0.5, 0.9]).to_string())
    print(f"slope at or below {SLOPE_FLOOR:g} (backwater and proxy left null): "
          f"{int(analyzed['slope_at_floor'].sum())}; "
          f"backwater_km > 50: {int((analyzed['backwater_km'] > 50).sum())}")
    print("\n=== Correlations (dams in dam_removal; log10 Pearson, Spearman) ===")
    print(cor.to_string(index=False))
    print("\n=== Selection bias (all removals) ===")
    print(bias.to_string(index=False))
    print("\n=== Collinearity with log10 dam drainage area (dams in dam_removal) ===")
    print(collinear.to_string(index=False))
    print(f"\nWrote {', '.join(names)} to {out_dir}")


if __name__ == "__main__":
    main()
