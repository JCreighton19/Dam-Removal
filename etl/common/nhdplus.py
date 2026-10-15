"""Shared helpers for locating extracted NHDPlus HR geodatabases."""

from __future__ import annotations

from pathlib import Path


def find_huc8_geodatabase(nhdplus_extracted_dir, huc8):
    """Locate the extracted NHDPlus HR geodatabase covering a HUC8.

    NHDPlus HR is distributed per-HUC8 where available, but some HUC8s
    only have a HUC4-level ("HU4") package covering their whole parent
    HUC4 region -- download_nhdplus.py falls back to this when no HU8
    package exists for a requested HUC8. This checks, in order:

      1. An exact HU8 package: NHDPLUS_H_<huc8>_HU8_GDB.gdb
      2. The HU4 fallback covering this HUC8's parent HUC4:
         NHDPLUS_H_<huc8[:4]>_HU4_GDB.gdb
      3. A looser search for any extracted .gdb whose name contains the
         HUC8 or its parent HUC4 code, in case the extraction directory
         wasn't named exactly like the zip.

    Raises FileNotFoundError, naming every path checked, if none exist.
    """
    nhdplus_extracted_dir = Path(nhdplus_extracted_dir)
    huc4 = huc8[:4]

    candidates = [
        nhdplus_extracted_dir
        / f"NHDPLUS_H_{huc8}_HU8_GDB"
        / f"NHDPLUS_H_{huc8}_HU8_GDB.gdb",
        nhdplus_extracted_dir
        / f"NHDPLUS_H_{huc4}_HU4_GDB"
        / f"NHDPLUS_H_{huc4}_HU4_GDB.gdb",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if nhdplus_extracted_dir.exists():
        for path in nhdplus_extracted_dir.rglob("*.gdb"):
            if huc8 in path.name or huc4 in path.name:
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
