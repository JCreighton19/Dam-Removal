"""
Shrink the national Watershed Boundary Dataset to the one layer the
pipeline reads (HUC8 polygons), to save space.

The national WBD GeoPackage holds every HUC level (HU2 to HU16) plus
other layers, but the pipeline only uses the HU8 layer (sample_random_pilot,
snap_features, load_core_tables). This rewrites the GeoPackage named by
the config's `wbd_source` in place with just that layer (via a .tmp file,
so a failure leaves the original untouched), and deletes the downloaded
zip (national_wbd.zip) next to it.

Run once after unzip_huc8 (its Slurm job runs it automatically). Running
it again on an already-slim file does nothing.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyogrio

from etl.common.config import load_config
from etl.common.io import require_file
from etl.common.wbd import find_hu8_layer


def slim_wbd(wbd_path):
    layers = [str(name) for name, _ in pyogrio.list_layers(wbd_path)]
    hu8 = find_hu8_layer(wbd_path)
    if layers == [hu8]:
        return "already slim"

    tmp = wbd_path.with_name(wbd_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    frame = pyogrio.read_dataframe(wbd_path, layer=hu8)
    with warnings.catch_warnings():
        # GDAL warns that the .tmp file isn't named .gpkg; it's renamed below.
        warnings.filterwarnings("ignore", message=".*non conformant file extension")
        pyogrio.write_dataframe(frame, tmp, layer=hu8, driver="GPKG")
    tmp.replace(wbd_path)
    return f"kept {hu8} ({len(frame):,} HUC8s), dropped {len(layers) - 1} other layers"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-case-study",
        default="pilot_19020401",
        help="Config whose wbd_source names the WBD GeoPackage.",
    )
    args = parser.parse_args()

    wbd_path = require_file(load_config(args.base_case_study).path("wbd_source"),
                            "WBD GeoPackage")
    before = wbd_path.stat().st_size / 1e9
    print(f"{wbd_path.name}: {slim_wbd(wbd_path)}")
    print(f"Size: {before:.1f} GB -> {wbd_path.stat().st_size / 1e9:.1f} GB")

    zip_path = wbd_path.parent / "national_wbd.zip"
    if zip_path.exists():
        size = zip_path.stat().st_size / 1e9
        zip_path.unlink()
        print(f"Deleted {zip_path.name} ({size:.1f} GB)")


if __name__ == "__main__":
    main()
