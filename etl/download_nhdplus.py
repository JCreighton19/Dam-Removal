from pathlib import Path
from urllib.parse import quote
import argparse
import csv
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path("/courses/DS5500.202710/teams/team4")
HUC4_FILE = ROOT / "data/processed/huc4_list.txt"
HUC8_FILE = ROOT / "data/processed/huc8_list.txt"
OUTPUT_DIR = ROOT / "data/raw/nhdplus"
MANIFEST_FILE = OUTPUT_DIR / "download_manifest.csv"
BUCKET_URL = "https://prd-tnm.s3.amazonaws.com/"
PREFIX = "StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/"
TIMEOUT = (30, 180)
CHUNK_SIZE = 1024 * 1024  # 1 MB

NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

HU4_PATTERN = re.compile(
    r"^NHDPLUS_H_(\d{4})_HU4(?:_(\d{8}))?_GDB\.zip$",
    re.IGNORECASE,
)
HU8_PATTERN = re.compile(
    r"^NHDPLUS_H_(\d{8})_HU8(?:_(\d{8}))?_GDB\.zip$",
    re.IGNORECASE,
)

MANIFEST_FIELDS = [
    "huc_code",
    "huc_level",
    "filename",
    "url",
    "size_bytes",
    "status",
    "message",
]


# --------------------------------------------------
# Read pilot watershed codes
# --------------------------------------------------

def read_codes(path, expected_length):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    codes = set()

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # Allows one code per line, including a simple CSV/header.
        matches = re.findall(r"\d+", line)
        for code in matches:
            if len(code) == expected_length:
                codes.add(code)

    if not codes:
        raise ValueError(
            f"No {expected_length}-digit codes found in {path}"
        )

    return sorted(codes)


# --------------------------------------------------
# List available current NHDPlus HR vector ZIPs
# --------------------------------------------------

def list_s3_files(session):
    print("Listing available NHDPlus HR packages...", flush=True)

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

            # Keep only the expected HU4/HU8 vector ZIP naming patterns.
            if not (HU4_PATTERN.fullmatch(filename)
                    or HU8_PATTERN.fullmatch(filename)):
                continue

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

    print(f"Found {len(files):,} candidate vector packages.", flush=True)
    return files


# --------------------------------------------------
# Choose the latest listed package for each HUC
# --------------------------------------------------

def latest_packages(files, pattern):
    packages = {}

    for item in files:
        match = pattern.fullmatch(item["filename"])
        if not match:
            continue

        code = match.group(1)
        date = match.group(2) or "00000000"
        existing = packages.get(code)

        if existing is None or date > existing["_date"]:
            packages[code] = {**item, "_date": date}

    return packages


# --------------------------------------------------
# Build a download plan
# --------------------------------------------------

def build_plan(huc4_codes, huc8_codes, files):
    hu4_packages = latest_packages(files, HU4_PATTERN)
    hu8_packages = latest_packages(files, HU8_PATTERN)
    plan = []
    missing = []

    for huc4 in huc4_codes:
        if huc4 in hu4_packages:
            package = hu4_packages[huc4]
            plan.append({
                **package,
                "huc_code": huc4,
                "huc_level": "HU4",
            })
            continue

        # No HU4 package: fall back only to pilot HU8 packages.
        fallback_codes = [
            code for code in huc8_codes
            if code.startswith(huc4)
        ]

        found_fallback = False

        for huc8 in fallback_codes:
            if huc8 in hu8_packages:
                plan.append({
                    **hu8_packages[huc8],
                    "huc_code": huc8,
                    "huc_level": "HU8",
                })
                found_fallback = True
            else:
                missing.append({
                    "huc_code": huc8,
                    "huc_level": "HU8",
                    "filename": "",
                    "url": "",
                    "size_bytes": 0,
                    "status": "missing",
                    "message": "No matching HU8 ZIP found in current listing",
                })

        if not found_fallback:
            missing.append({
                "huc_code": huc4,
                "huc_level": "HU4",
                "filename": "",
                "url": "",
                "size_bytes": 0,
                "status": "missing",
                "message": (
                    "No HU4 package and no matching pilot HU8 package"
                ),
            })

    # Avoid duplicate package downloads.
    unique = {item["filename"]: item for item in plan}
    plan = sorted(unique.values(), key=lambda x: x["filename"])

    return plan, missing


# --------------------------------------------------
# Download and validate one ZIP
# --------------------------------------------------

def download_package(session, item):
    destination = OUTPUT_DIR / item["filename"]
    partial = OUTPUT_DIR / (item["filename"] + ".part")

    if destination.exists() and zipfile.is_zipfile(destination):
        print(f"Already valid; skipping: {item['filename']}", flush=True)
        return {
            **item,
            "status": "skipped_existing",
            "message": "Existing ZIP passed validation",
        }

    # Remove an invalid or incomplete prior download.
    if destination.exists():
        destination.unlink()

    if partial.exists():
        partial.unlink()

    print(
        f"Downloading {item['filename']} "
        f"({item['size_bytes'] / 1e9:.2f} GB)...",
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
            raise ValueError(
                "Downloaded file is not a valid ZIP archive"
            )

        actual_size = partial.stat().st_size
        partial.replace(destination)
        elapsed = time.time() - start

        print(
            f"Completed: {item['filename']} "
            f"({actual_size / 1e9:.2f} GB, {elapsed:.0f}s)",
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
            f"FAILED: {item['filename']}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        return {
            **item,
            "status": "failed",
            "message": str(exc),
        }


# --------------------------------------------------
# Write manifest after each result
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="List planned downloads without downloading files",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    huc4_codes = read_codes(HUC4_FILE, 4)
    huc8_codes = read_codes(HUC8_FILE, 8)

    print(f"Pilot HUC4 codes: {len(huc4_codes)}", flush=True)
    print(f"Pilot HUC8 codes: {len(huc8_codes)}", flush=True)

    with requests.Session() as session:
        files = list_s3_files(session)
        plan, missing = build_plan(
            huc4_codes,
            huc8_codes,
            files,
        )

        print(f"\nPackages selected: {len(plan)}", flush=True)

        for item in plan:
            print(
                f"  {item['huc_level']} {item['huc_code']}: "
                f"{item['filename']} "
                f"({item['size_bytes'] / 1e9:.2f} GB)",
                flush=True,
            )

        total_gb = sum(x["size_bytes"] for x in plan) / 1e9
        print(f"Total listed size: {total_gb:.2f} GB", flush=True)

        if args.plan_only:
            rows = [
                {
                    **item,
                    "status": "planned",
                    "message": "Plan-only; not downloaded",
                }
                for item in plan
            ] + missing
            write_manifest(rows)
            print("Plan-only run finished.", flush=True)
            return

        rows = list(missing)
        write_manifest(rows)

        for item in plan:
            result = download_package(session, item)
            rows.append(result)
            write_manifest(rows)

    print("\nDownload summary", flush=True)

    for status in (
        "downloaded",
        "skipped_existing",
        "failed",
        "missing",
    ):
        count = sum(row["status"] == status for row in rows)
        print(f"  {status}: {count}", flush=True)

    print(f"Manifest: {MANIFEST_FILE}", flush=True)

    if any(row["status"] == "failed" for row in rows):
        raise RuntimeError(
            "One or more downloads failed. Check the manifest and logs."
        )


if __name__ == "__main__":
    main()