"""Shared helpers for locating extracted NHDPlus HR geodatabases.

unzip_nhdplus.py keeps each package as a slim GeoPackage
(<package>/<package>.gpkg) holding only the layers the pipeline reads
(SLIM_LAYERS). A full FileGDB (<package>/<package>.gdb) is still found,
so older extractions keep working.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pyogrio

# Column names as the rest of the pipeline spells them. Newer NHDPlus HR
# releases (e.g. NHDPLUS_H_0106_HU4_20230713_GDB) use lowercase field names
# (nhdplusid, fromnhdpid, ...), so reads normalize to these spellings.
# The only NHDPlus HR layers the pipeline reads.
SLIM_LAYERS = ("NHDFlowline", "NHDPlusFlow", "NHDPlusCatchment", "NHDPlusFlowlineVAA")

# Extensions of an extracted package, preferred first.
PACKAGE_EXTENSIONS = (".gpkg", ".gdb")

CANONICAL_COLUMNS = (
    "NHDPlusID",
    "LengthKM",
    "FromNHDPID",
    "ToNHDPID",
    "ReachCode",
    "GNIS_Name",
    "FType",
    "FCode",
    "FlowDir",
    "TotDASqKm",
    "Slope",
)


def find_huc8_geodatabase(nhdplus_extracted_dir, huc8):
    """Locate the extracted NHDPlus HR geodatabase covering a HUC8.

    NHDPlus HR is distributed per-HUC8 where available, but some HUC8s
    only have a HUC4-level ("HU4") package covering their whole parent
    HUC4 region -- download_nhdplus.py falls back to this when no HU8
    package exists for a requested HUC8. This checks, in order:

      1. An exact HU8 package: NHDPLUS_H_<huc8>_HU8_GDB.gpkg (or .gdb)
      2. The HU4 fallback covering this HUC8's parent HUC4:
         NHDPLUS_H_<huc8[:4]>_HU4_GDB.gpkg (or .gdb)
      3. A looser search for any extracted .gpkg/.gdb whose name contains the
         HUC8 or its parent HUC4 code, in case the extraction directory
         wasn't named exactly like the zip.

    Raises FileNotFoundError, naming every path checked, if none exist.
    """
    nhdplus_extracted_dir = Path(nhdplus_extracted_dir)
    huc4 = huc8[:4]

    candidates = [
        nhdplus_extracted_dir / package / f"{package}{extension}"
        for package in (f"NHDPLUS_H_{huc8}_HU8_GDB", f"NHDPLUS_H_{huc4}_HU4_GDB")
        for extension in PACKAGE_EXTENSIONS
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if nhdplus_extracted_dir.exists():
        for extension in PACKAGE_EXTENSIONS:
            for path in sorted(nhdplus_extracted_dir.rglob(f"*{extension}")):
                # Match whole "_"-separated tokens so a release date such as
                # _20230713_ doesn't match HUC4 0713.
                tokens = path.stem.split("_")
                if huc8 in tokens or huc4 in tokens:
                    return path

    raise FileNotFoundError(
        f"No extracted NHDPlus HR geodatabase found for HUC8 {huc8} "
        f"(checked for an HU8 package and an HU4 fallback covering "
        f"HUC4 {huc4}) under {nhdplus_extracted_dir}. Checked: "
        f"{[str(c) for c in candidates]}"
    )


def is_hu4_fallback(gdb_path, huc8):
    """True if the located geodatabase is an HU4 fallback, not an exact HU8 match."""
    return f"_{huc8}_HU8_GDB" not in Path(gdb_path).name


def normalize_nhdplus_columns(frame):
    """Rename NHDPlus columns to their canonical spelling, ignoring case."""
    canonical = {name.lower(): name for name in CANONICAL_COLUMNS}
    renames = {
        col: canonical[col.lower()]
        for col in frame.columns
        if col.lower() in canonical and col != canonical[col.lower()]
    }
    return frame.rename(columns=renames)


def read_nhdplus_layer(gdb_path, layer, columns=None, **kwargs):
    """Read an NHDPlus HR layer with canonical column names.

    `columns` may use canonical names regardless of how the geodatabase
    spells them. Extra kwargs (where, ignore_geometry, ...) pass through to
    geopandas.read_file.
    """
    if columns is not None:
        fields = pyogrio.read_info(gdb_path, layer=layer)["fields"]
        available = {str(field).lower(): str(field) for field in fields}
        missing = [col for col in columns if col.lower() not in available]
        if missing:
            raise ValueError(
                f"Layer '{layer}' in {gdb_path} is missing columns {missing}. "
                f"Available: {sorted(available.values())}"
            )
        columns = [available[col.lower()] for col in columns]

    frame = gpd.read_file(gdb_path, layer=layer, columns=columns, **kwargs)
    return normalize_nhdplus_columns(frame)
