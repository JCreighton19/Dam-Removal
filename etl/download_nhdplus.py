from pathlib import Path
from urllib.parse import quote
import csv
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.common.config import default_case_study, load_config


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path("/courses/DS5500.202710/teams/team4")
OUTPUT_DIR = ROOT / "data/raw/nhdplus"
MANIFEST_FILE = OUTPUT_DIR / "download_manifest.csv"
BUCKET_URL = "https://prd-tnm.s3.amazonaws.com/"
PREFIX = "StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/"
TIMEOUT = (30, 180)
CHUNK_SIZE = 1024 * 1024
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

HU8_PATTERN = re.compile(
    r"^NHDPLUS_H_(\d{8})_HU8(?:_(\d{8}))?_GDB\.zip$",
    re.IGNORECASE,
)

HU4_PATTERN = re.compile(
    r"^NHDPLUS_H_(\d{4})_HU4(?:_(\d{8}))?_GDB\.zip$",
    re.IGNORECASE,
)

MANIFEST_FIELDS = [
    "huc_code",
    "filename",
    "url",
    "size_bytes",
    "status",
    "message",
]


# --------------------------------------------------
# Read HUC8 codes from the pilot station CSV
# --------------------------------------------------

def read_pilot_huc8s(path):
    if not path.exists():
        raise FileNotFoundError(f"Station CSV not found: {path}")

    huc8_codes = set()
    station_count = 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "source_huc8" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain a 'source_huc8' column. "
                f"Columns found: {reader.fieldnames}"
            )

        for row in reader:
            station_count += 1
            value = (row.get("source_huc8") or "").strip()

            if not value:
                continue

            # Handle values represented as floats, e.g. 3010201.0.
            if value.endswith(".0"):
                value = value[:-2]

            # Remove formatting characters, then restore leading zeros
            # if the source CSV stored the code as a number.
            code = re.sub(r"\D", "", value)

            if 1 <= len(code) <= 8:
                code = code.zfill(8)

            if len(code) == 8:
                huc8_codes.add(code)
            else:
                print(
                    f"WARNING: Ignoring invalid HUC8 value: {value!r}",
                    flush=True,
                )

    if not huc8_codes:
        raise ValueError("No valid HUC8 codes found in the pilot CSV.")

    print(f"Pilot station rows: {station_count:,}", flush=True)
    print(f"Unique pilot HUC8 watersheds: {len(huc8_codes):,}", flush=True)

    for code in sorted(huc8_codes):
        print(f"  {code}", flush=True)

    return sorted(huc8_codes)


# --------------------------------------------------
# List available NHDPlus HR vector packages
# --------------------------------------------------


def list_s3_files(session):
    print("\nListing available NHDPlus HR packages...", flush=True)

    params = {
        "list-type": "2",
        "prefix": PREFIX,
    }

    files = []

    while True:
        response = session.get(
            BUCKET_URL,
            params=params,
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)

        for item in root.findall("s3:Contents", NS):
            key = item.findtext("s3:Key", namespaces=NS, default="")
            size = item.findtext(
                "s3:Size", namespaces=NS, default="0"
            )
            filename = key.rsplit("/", 1)[-1]

            if (
                HU8_PATTERN.fullmatch(filename)
                or HU4_PATTERN.fullmatch(filename)
            ):
                files.append({
                    "filename": filename,
                    "url": BUCKET_URL + quote(key, safe="/"),
                    "size_bytes": int(size),
                })

        token = root.findtext(
            "s3:NextContinuationToken",
            namespaces=NS,
            default="",
        )

        if not token:
            break

        params["continuation-token"] = token

    hu8_count = sum(
        HU8_PATTERN.fullmatch(item["filename"]) is not None
        for item in files
    )
    hu4_count = sum(
        HU4_PATTERN.fullmatch(item["filename"]) is not None
        for item in files
    )

    print(f"Found {hu8_count:,} HU8 packages.", flush=True)
    print(f"Found {hu4_count:,} HU4 packages.", flush=True)

    return files


# --------------------------------------------------
# Match packages to pilot HUC8s
# --------------------------------------------------


def select_packages(huc8_codes, files):
    available_hu8 = {}
    available_hu4 = {}

    for item in files:
        filename = item["filename"]

        match_hu8 = HU8_PATTERN.fullmatch(filename)
        if match_hu8:
            code = match_hu8.group(1)
            date = match_hu8.group(2) or "00000000"

            if (
                code not in available_hu8
                or date > available_hu8[code]["_date"]
            ):
                available_hu8[code] = {**item, "_date": date}

        match_hu4 = HU4_PATTERN.fullmatch(filename)
        if match_hu4:
            code = match_hu4.group(1)
            date = match_hu4.group(2) or "00000000"

            if (
                    code not in available_hu4
                    or date > available_hu4[code]["_date"]
            ):
                available_hu4[code] = {
                    **item,
                    "_date": date,
                }

    selected = []
    missing = []

    # Group requested HUC8s by their HUC4 prefix.
    huc4_groups = {}
    for huc8 in huc8_codes:
        huc4_groups.setdefault(huc8[:4], []).append(huc8)

    for huc4, requested_huc8s in sorted(huc4_groups.items()):
        missing_huc8s = [
            huc8 for huc8 in requested_huc8s
            if huc8 not in available_hu8
        ]

        # If any requested HUC8 is missing, prefer one HU4 package
        # for this entire HUC4, rather than mixing overlapping files.
        if missing_huc8s:
            if huc4 in available_hu4:
                item = available_hu4[huc4]

                selected.append({
                    **item,
                    "huc_code": huc4,
                })

                print(
                    f"Fallback to HU4 {huc4}: covers requested "
                    f"HUC8(s) {', '.join(requested_huc8s)}",
                    flush=True,
                )
            else:
                for huc8 in missing_huc8s:
                    missing.append({
                        "huc_code": huc8,
                        "filename": "",
                        "url": "",
                        "size_bytes": 0,
                        "status": "missing",
                        "message": (
                            f"No HU8 package or HU4 fallback "
                            f"for HUC4 {huc4}"
                        ),
                    })

                # Download any requested HU8s that are available.
                for huc8 in requested_huc8s:
                    if huc8 in available_hu8:
                        selected.append({
                            **available_hu8[huc8],
                            "huc_code": huc8,
                        })
        else:
            # All requested HU8 packages for this HUC4 are available.
            for huc8 in requested_huc8s:
                selected.append({
                    **available_hu8[huc8],
                    "huc_code": huc8,
                })

    return selected, missing


# --------------------------------------------------
# Download and validate a package
# --------------------------------------------------

def download_package(session, item):
    destination = OUTPUT_DIR / item["filename"]
    partial = OUTPUT_DIR / (item["filename"] + ".part")

    if destination.exists() and zipfile.is_zipfile(destination):
        print(f"Already valid; skipping {item['filename']}", flush=True)
        return {
            **item,
            "status": "skipped_existing",
            "message": "Existing ZIP passed validation",
        }

    if destination.exists():
        destination.unlink()

    if partial.exists():
        partial.unlink()

    print(
        f"\nDownloading {item['filename']} "
        f"({item['size_bytes'] / 1e9:.2f} GB)",
        flush=True,
    )

    start = time.time()

    try:
        with session.get(
            item["url"],
            stream=True,
            timeout=TIMEOUT,
        ) as response:
            response.raise_for_status()

            with partial.open("wb") as output:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):
                    if chunk:
                        output.write(chunk)

        if not zipfile.is_zipfile(partial):
            raise ValueError("Downloaded file is not a valid ZIP archive")

        actual_size = partial.stat().st_size
        partial.replace(destination)

        print(
            f"Completed {item['filename']} "
            f"({actual_size / 1e9:.2f} GB, "
            f"{time.time() - start:.0f}s)",
            flush=True,
        )

        return {
            **item,
            "size_bytes": actual_size,
            "status": "downloaded",
            "message": "ZIP validation passed",
        }

    except Exception as exc:
        if partial.exists():
            partial.unlink()

        print(
            f"FAILED {item['filename']}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        return {
            **item,
            "status": "failed",
            "message": str(exc),
        }


# --------------------------------------------------
# Write manifest
# --------------------------------------------------

def write_manifest(rows):
    with MANIFEST_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in MANIFEST_FIELDS
            })


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Show planned downloads without downloading",
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    args = parser.parse_args()
    cfg = load_config(args.case_study)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    huc8_codes = read_pilot_huc8s(cfg.wqp_results_dir / "pilot_station_selection.csv")

    with requests.Session() as session:
        files = list_s3_files(session)
        selected, missing = select_packages(huc8_codes, files)

        print("\nDownload plan", flush=True)
        print(f"Packages selected: {len(selected)}", flush=True)
        print(f"HUC8s still missing: {len(missing)}", flush=True)

        total_bytes = sum(item["size_bytes"] for item in selected)
        print(f"Total listed download size: {total_bytes / 1e9:.2f} GB",
              flush=True)

        for item in selected:
            print(
                f"  {item['huc_code']}: {item['filename']} "
                f"({item['size_bytes'] / 1e9:.2f} GB)",
                flush=True,
            )

        rows = list(missing)

        if args.plan_only:
            rows.extend({
                **item,
                "status": "planned",
                "message": "Plan-only; not downloaded",
            } for item in selected)

            write_manifest(rows)
            print(f"\nPlan written to {MANIFEST_FILE}", flush=True)
            return

        write_manifest(rows)

        for item in selected:
            result = download_package(session, item)
            rows.append(result)
            write_manifest(rows)

    print("\nDownload summary", flush=True)

    for status in ("downloaded", "skipped_existing", "failed", "missing"):
        count = sum(row.get("status") == status for row in rows)
        print(f"  {status}: {count}", flush=True)

    print(f"Manifest: {MANIFEST_FILE}", flush=True)

    if any(row.get("status") == "failed" for row in rows):
        raise RuntimeError("One or more downloads failed; inspect the manifest.")


if __name__ == "__main__":
    main()