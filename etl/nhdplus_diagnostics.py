"""
Stage 1 of the reach-matching rework: check, package by package, the
NHDPlus HR relationships station and dam matching will rely on
(etl/common/water_features.py). Reads attribute tables only; no geometry.

For the case study's HUC8s (or --hu4 / --huc8 subsets), each extracted
package (an HU8 package, or the HU4 package covering several HUC8s) is
checked once:

- artificial paths (FType 558) with and without WBArea_Permanent_Identifier,
  and whether that identifier resolves to an NHDArea or NHDWaterbody;
- water polygons by layer and FType: how many have artificial paths, and
  their NHDPlusBurnWaterbody on/off-network flag;
- catchments without a flowline, matchable flowlines (460/558) without a
  catchment, VAA row or valid TotDASqKm;
- NHDPlusFlow ids that aren't flowlines of the package;
- packages missing the water layers (run download_streams.slurm, which
  adds them, before relying on these results).

Outputs in --output-dir (default
data/processed/nhdplus_diagnostics/<case_study>/):

    huc8_coverage.csv     one row per HUC8: its package, level, missing layers
    package_summary.csv   one row per package: counts and exception totals
    polygon_types.csv     per package, layer and FType: polygons, linked, on-network
    exceptions.csv        problem records (first --max-exceptions per package
                          and kind; totals are in package_summary.csv)

Existing pipeline outputs are not touched.
"""

from pathlib import Path
import argparse
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyogrio

from etl.common import water_features as wf
from etl.common.config import default_case_study, load_config
from etl.common.nhdplus import (
    CORE_LAYERS,
    WATER_LAYERS,
    find_huc8_geodatabase,
    missing_slim_layers,
    normalize_nhdplus_columns,
)

# Columns read from each layer (canonical spelling; any case matches).
# The first column of each list is required; the rest are read if present.
LAYER_COLUMNS = {
    "NHDFlowline": ["NHDPlusID", "FType", "FCode", "WBArea_Permanent_Identifier",
                    "Permanent_Identifier", "GNIS_Name"],
    "NHDArea": ["Permanent_Identifier", "NHDPlusID", "FType", "FCode"],
    "NHDWaterbody": ["Permanent_Identifier", "NHDPlusID", "FType", "FCode"],
    "NHDPlusBurnWaterbody": ["NHDPlusID", "OnOffNet", "SourceFC"],
    "NHDPlusCatchment": ["NHDPlusID", "SourceFC"],
    "NHDPlusFlowlineVAA": ["NHDPlusID", "TotDASqKm"],
    "NHDPlusFlow": ["FromNHDPID", "ToNHDPID"],
}

PACKAGE_PATTERN = re.compile(r"NHDPLUS_H_(\d{4}|\d{8})_HU(4|8)", re.IGNORECASE)


def package_info(path):
    """(package name, level "HU4"/"HU8", HUC4) from an extracted package path."""
    match = PACKAGE_PATTERN.search(Path(path).name)
    if not match:
        return Path(path).stem, None, None
    return Path(path).stem, f"HU{match.group(2)}", match.group(1)[:4]


def read_attributes(path, layer):
    """The layer's LAYER_COLUMNS that exist (canonical names), no geometry.
    Returns (frame, missing columns). A missing layer reads as empty."""
    wanted = LAYER_COLUMNS[layer]
    layers = {str(name).lower(): str(name) for name, _ in pyogrio.list_layers(path)}
    if layer.lower() not in layers:
        return pd.DataFrame(columns=wanted), wanted
    fields = {str(f).lower(): str(f) for f in pyogrio.read_info(path, layer=layers[layer.lower()])["fields"]}
    present = [fields[c.lower()] for c in wanted if c.lower() in fields]
    missing = [c for c in wanted if c.lower() not in fields]
    frame = pyogrio.read_dataframe(path, layer=layers[layer.lower()], columns=present,
                                   read_geometry=False)
    frame = normalize_nhdplus_columns(pd.DataFrame(frame))
    for column in missing:
        frame[column] = pd.NA
    return frame, missing


def check_package(path, max_exceptions):
    """(summary row, polygon_types, exceptions) for one package."""
    name, level, huc4 = package_info(path)
    row = {"package": name, "level": level, "hu4": huc4, "path": str(path)}
    missing_layers = missing_slim_layers(path)
    row["missing_layers"] = ";".join(missing_layers) or None
    row["has_water_layers"] = not any(layer in WATER_LAYERS for layer in missing_layers)
    if any(layer in CORE_LAYERS for layer in missing_layers):
        row["status"] = "missing_core_layers"
        return row, None, None

    tables, missing_columns = {}, []
    for layer in LAYER_COLUMNS:
        tables[layer], missing = read_attributes(path, layer)
        if layer.lower() not in {m.lower() for m in missing_layers}:
            missing_columns += [f"{layer}.{column}" for column in missing]
    row["missing_columns"] = ";".join(missing_columns) or None

    counts, polygon_types, exceptions = wf.relationship_checks(
        tables["NHDFlowline"], tables["NHDArea"], tables["NHDWaterbody"],
        tables["NHDPlusCatchment"], tables["NHDPlusFlowlineVAA"], tables["NHDPlusFlow"],
        tables["NHDPlusBurnWaterbody"])
    row.update(counts)
    row["status"] = "checked" if row["has_water_layers"] else "checked_without_water_layers"
    keep = exceptions.groupby("kind", group_keys=False).head(max_exceptions)
    for frame in (polygon_types, keep):
        frame.insert(0, "package", name)
        frame.insert(1, "hu4", huc4)
    return row, polygon_types, keep


def choose_huc8s(cfg, huc8s, hu4s):
    chosen = list(cfg.huc8s)
    if huc8s:
        chosen = [h for h in chosen if h in set(huc8s)]
    if hu4s:
        chosen = [h for h in chosen if h[:4] in set(hu4s)]
    return chosen


def print_summary(summary, coverage):
    print("\n=== HUC8 coverage ===")
    print(coverage["package_status"].value_counts(dropna=False).to_string())
    print("\n=== Packages ===")
    print(summary["status"].value_counts(dropna=False).to_string())
    checked = summary[summary["status"] == "checked"]
    if checked.empty:
        print("No package with water layers was checked.")
        return
    by_hu4 = checked.groupby("hu4").agg(
        packages=("package", "size"),
        artificial_paths=("n_artificial_path_558", "sum"),
        with_polygon_id=("n_ap_with_polygon_id", "sum"),
        unresolved=("n_ap_polygon_unresolved", "sum"),
        to_nhdarea=("n_ap_to_nhdarea", "sum"),
        to_nhdwaterbody=("n_ap_to_nhdwaterbody", "sum"),
        matchable_without_catchment=("n_matchable_without_catchment", "sum"))
    by_hu4["pct_with_polygon_id"] = (100 * by_hu4["with_polygon_id"]
                                     / by_hu4["artificial_paths"]).round(2)
    print("\n=== Artificial paths and catchments by HUC4 ===")
    print(by_hu4.to_string())
    exc = [c for c in summary.columns if c.startswith("exc_")]
    if exc:
        print("\n=== Exception totals (all checked packages) ===")
        print(checked[exc].sum().sort_values(ascending=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--hu4", nargs="+", help="Only the case study's HUC8s in these HUC4s.")
    parser.add_argument("--huc8", nargs="+", help="Only these case-study HUC8s.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-exceptions", type=int, default=500,
                        help="Exception rows kept per package and kind (default 500).")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    output_dir = args.output_dir or (cfg.project_root / "data" / "processed"
                                     / "nhdplus_diagnostics" / cfg.name)
    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    huc8s = choose_huc8s(cfg, args.huc8, args.hu4)
    print(f"Case study: {cfg.name}; HUC8s: {len(huc8s)}", flush=True)
    if not huc8s:
        raise SystemExit("No case-study HUC8 matches --hu4/--huc8.")

    coverage, packages = [], {}
    for huc8 in huc8s:
        try:
            path = find_huc8_geodatabase(nhd_dir, huc8)
        except FileNotFoundError:
            coverage.append({"huc8": huc8, "hu4": huc8[:4], "package": None, "level": None,
                             "package_status": "no_extracted_package"})
            continue
        name, level, _ = package_info(path)
        packages[name] = path
        coverage.append({"huc8": huc8, "hu4": huc8[:4], "package": name, "level": level,
                         "package_status": "found"})

    rows, polygon_types, exceptions = [], [], []
    for number, (name, path) in enumerate(sorted(packages.items()), start=1):
        print(f"[{number}/{len(packages)}] {name}", flush=True)
        try:
            row, types, exc = check_package(path, args.max_exceptions)
        except Exception as error:
            print(f"  FAILED: {error}", flush=True)
            row, types, exc = {"package": name, "path": str(path),
                               "status": f"failed: {error}"}, None, None
        rows.append(row)
        if types is not None:
            polygon_types.append(types)
            exceptions.append(exc)
        print(f"  {row['status']}; missing layers: {row.get('missing_layers')}", flush=True)

    summary = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage)
    status = summary.set_index("package")["status"] if len(summary) else pd.Series(dtype=str)
    coverage["package_check"] = coverage["package"].map(status)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "huc8_coverage.csv", index=False)
    summary.to_csv(output_dir / "package_summary.csv", index=False)
    (pd.concat(polygon_types, ignore_index=True) if polygon_types else pd.DataFrame()) \
        .to_csv(output_dir / "polygon_types.csv", index=False)
    (pd.concat(exceptions, ignore_index=True) if exceptions else pd.DataFrame()) \
        .to_csv(output_dir / "exceptions.csv", index=False)

    print_summary(summary, coverage)
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
