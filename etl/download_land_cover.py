from pathlib import Path
import logging
import re

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DAM_CSV = Path("data/raw/dams/ARDamRemovalList_Figshare_Mar2026.csv")
WBD_GPKG = Path("data/raw/wbd/WBD_National_GPKG.gpkg")
OUTPUT_DIR = Path("data/raw/nlcd")
RASTER_DIR = OUTPUT_DIR / "rasters"

# Change accordingly
PILOT_HUC8 = "19020401"
YEARS_BEFORE = 3
YEARS_AFTER = 3

# Current Annual NLCD Collection 1 versions, newest first.
COLLECTION = "c1"
VERSIONS = [2, 1, 0]

# Annual NLCD land-cover products currently cover CONUS.
# This is a coarse geographic filter, not a substitute for checking
# actual raster coverage.
CONUS_BOUNDS = (-125.0, 24.0, -66.0, 50.0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Prepare dams and watershed geometries
# ---------------------------------------------------------------------

def load_dams_and_watersheds():
    dams = pd.read_csv(DAM_CSV, dtype={"HUC8": str})
    wbd = gpd.read_file(WBD_GPKG, layer="WBDHU8")

    required_dam_columns = {
        "AR_ID", "HUC8", "Year_Removed", "Latitude", "Longitude"
    }
    missing = required_dam_columns - set(dams.columns)
    if missing:
        raise ValueError(f"Missing dam columns: {sorted(missing)}")

    if not {"huc8", "name", "geometry"}.issubset(wbd.columns):
        raise ValueError("WBDHU8 must contain huc8, name, and geometry.")

    dams["Year_Removed"] = pd.to_numeric(
        dams["Year_Removed"], errors="coerce"
    )
    dams["HUC8"] = dams["HUC8"].astype("string").str.strip()

    wbd = wbd.to_crs("EPSG:4326").copy()
    wbd["huc8"] = wbd["huc8"].astype(str).str.zfill(8)

    # Use spatial matching where coordinates are available.
    valid_xy = (
        pd.to_numeric(dams["Longitude"], errors="coerce").between(-180, 180)
        & pd.to_numeric(dams["Latitude"], errors="coerce").between(-90, 90)
    )

    dam_points = dams.loc[valid_xy].copy()
    dam_points["Longitude"] = pd.to_numeric(dam_points["Longitude"])
    dam_points["Latitude"] = pd.to_numeric(dam_points["Latitude"])

    dam_gdf = gpd.GeoDataFrame(
        dam_points,
        geometry=gpd.points_from_xy(
            dam_points["Longitude"], dam_points["Latitude"]
        ),
        crs="EPSG:4326",
    )

    spatial = gpd.sjoin(
        dam_gdf,
        wbd[["huc8", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    # Start with the source HUC8 field, then prefer spatially matched
    # HUC8 identifiers where the join succeeded.
    huc_by_dam = dams.set_index("AR_ID")["HUC8"].to_dict()
    for _, row in spatial.iterrows():
        if pd.notna(row.get("huc8")):
            huc_by_dam[row["AR_ID"]] = str(row["huc8"]).zfill(8)

    dams["huc8_resolved"] = dams["AR_ID"].map(huc_by_dam)

    # If the source field contains watershed NAMES rather than codes,
    # resolve those names against WBDHU8.
    name_to_huc = (
        wbd[["name", "huc8"]]
        .dropna()
        .drop_duplicates("name")
        .set_index("name")["huc8"]
        .to_dict()
    )

    def resolve_huc(value):
        if pd.isna(value):
            return None

        value = str(value).strip()

        if re.fullmatch(r"\d{8}", value):
            return value

        return name_to_huc.get(value)

    dams["huc8_resolved"] = dams["huc8_resolved"].map(resolve_huc)

    dams = dams[
        dams["Year_Removed"].notna()
        & dams["huc8_resolved"].notna()
    ].copy()

    dams["Year_Removed"] = dams["Year_Removed"].astype(int)

    # Dissolve duplicate watershed polygons into one geometry per HUC8.
    watersheds = (
        wbd[["huc8", "geometry"]]
        .dissolve(by="huc8")
    )

    return dams, watersheds


# ---------------------------------------------------------------------
# Build the download manifest
# ---------------------------------------------------------------------

def build_manifest(dams):
    requests = set()

    for row in dams[
        ["AR_ID", "huc8_resolved", "Year_Removed"]
    ].itertuples(index=False, name=None):
        dam_id, huc8, removal_year = row

        for year in range(
            removal_year - YEARS_BEFORE,
            removal_year + YEARS_AFTER + 1,
        ):
            requests.add((dam_id, huc8, year))

    manifest = pd.DataFrame(
        sorted(requests),
        columns=["AR_ID", "HUC8", "Land_Cover_Year"],
    )

    manifest.to_csv(OUTPUT_DIR / "download_manifest.csv", index=False)

    return manifest


# ---------------------------------------------------------------------
# Locate the remote annual land-cover raster
# ---------------------------------------------------------------------

def mosaic_url(year, version):
    filename = (
        f"Annual_NLCD_LndCov_{year}_CU_C1V{version}.tif"
    )

    return (
        "https://usgs-landcover.s3.us-west-2.amazonaws.com/"
        f"annual-nlcd/{COLLECTION}/v{version}/cu/mosaic/{filename}"
    )


def open_nlcd_raster(year):
    # Requester-pays is required by the official AWS distribution.
    # GDAL/rasterio must support this configuration for remote reads.
    env = rasterio.Env(
        AWS_REQUEST_PAYER="requester",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    )
    env.__enter__()

    for version in VERSIONS:
        url = mosaic_url(year, version)

        try:
            src = rasterio.open(url)
            logger.info("Opened NLCD %s (Collection 1, version %s)", year, version)
            return env, src, version
        except (RasterioIOError, Exception) as exc:
            logger.warning(
                "Could not open NLCD year %s version %s: %s",
                year, version, exc,
            )

    env.__exit__(None, None, None)
    raise RuntimeError(
        f"No accessible CONUS NLCD mosaic found for year {year}. "
        "The year may be unavailable, or remote AWS access may have failed."
    )


# ---------------------------------------------------------------------
# Clip a remote raster to one watershed
# ---------------------------------------------------------------------

def download_clip(huc8, year, watershed):
    out_dir = RASTER_DIR / huc8
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"NLCD_LandCover_{year}_{huc8}.tif"

    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Already exists: %s", output_path)
        return str(output_path), "already_exists", ""

    # Fast approximate geographic check. Do not try CONUS data for
    # clearly non-CONUS watersheds.
    bounds = watershed.to_crs("EPSG:4326").total_bounds
    minx, miny, maxx, maxy = bounds
    west, south, east, north = CONUS_BOUNDS

    if maxx < west or minx > east or maxy < south or miny > north:
        return "", "unsupported_region", (
            "Outside approximate CONUS coverage; current Annual NLCD "
            "land-cover mosaics do not cover this watershed."
        )

    env = None
    src = None

    try:
        env, src, version = open_nlcd_raster(year)

        # Reproject the watershed polygon to the raster CRS.
        geom = watershed.to_crs(src.crs)
        shapes = [
            geom.geometry.iloc[0].__geo_interface__
        ]

        # Read only the crop window from the remote raster where possible.
        clipped, transform = mask(
            src,
            shapes,
            crop=True,
            filled=True,
            nodata=src.nodata if src.nodata is not None else 250,
        )

        if clipped.size == 0:
            return "", "empty_clip", "Clipping returned no raster cells."

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
            compress="LZW",
            tiled=True,
            nodata=src.nodata if src.nodata is not None else 250,
        )

        # Write to a temporary file first so interrupted downloads
        # do not leave a seemingly valid final output.
        temp_path = output_path.with_suffix(".tmp.tif")

        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(clipped)

        temp_path.replace(output_path)

        logger.info("Saved %s", output_path)
        return str(output_path), "downloaded", f"Collection 1 version {version}"

    except Exception as exc:
        logger.exception("Failed HUC8 %s, year %s", huc8, year)
        return "", "failed", str(exc)

    finally:
        if src is not None:
            src.close()
        if env is not None:
            env.__exit__(None, None, None)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_DIR.mkdir(parents=True, exist_ok=True)

    dams, watersheds = load_dams_and_watersheds()
    dams = dams[dams["huc8_resolved"] == PILOT_HUC8].copy()
    watersheds = watersheds.loc[
        watersheds.index.astype(str) == PILOT_HUC8
        ]

    manifest = build_manifest(dams)

    print(f"Eligible dams: {dams['AR_ID'].nunique():,}")
    print(f"Unique HUC8 watersheds: {manifest['HUC8'].nunique():,}")
    print(f"Dam/HUC8/year requests: {len(manifest):,}")

    # Download each unique HUC8/year only once, even if multiple dams
    # in the same watershed have overlapping time windows.
    unique_requests = (
        manifest[["HUC8", "Land_Cover_Year"]]
        .drop_duplicates()
        .sort_values(["HUC8", "Land_Cover_Year"])
    )

    results = []

    for row in unique_requests.itertuples(index=False):
        huc8 = row.HUC8
        year = int(row.Land_Cover_Year)

        if huc8 not in watersheds.index:
            results.append({
                "HUC8": huc8,
                "Land_Cover_Year": year,
                "status": "missing_geometry",
                "output_path": "",
                "details": "No matching WBD watershed geometry.",
            })
            continue

        geometry = watersheds.loc[[huc8]]

        output_path, status, details = download_clip(
            huc8, year, geometry
        )

        results.append({
            "HUC8": huc8,
            "Land_Cover_Year": year,
            "status": status,
            "output_path": output_path,
            "details": details,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "download_results.csv", index=False
    )

    print("\nDownload summary:")
    print(results_df["status"].value_counts(dropna=False).to_string())
    print(f"\nManifest: {OUTPUT_DIR / 'download_manifest.csv'}")
    print(f"Results:  {OUTPUT_DIR / 'download_results.csv'}")
    print(f"Rasters:  {RASTER_DIR}")


if __name__ == "__main__":
    main()