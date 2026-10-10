from pathlib import Path
import argparse
import re
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HUC_FILE = ROOT / "data/processed/huc8_list.txt"
OUTPUT_DIR = ROOT / "data/raw/wqp/stations"
MANIFEST = ROOT / "data/raw/wqp/station_manifest.csv"

BASE_URL = "https://www.waterqualitydata.us/data/Station/search"


def read_hucs(path):
    """Extract unique eight-digit HUC codes, preserving order."""
    hucs = []

    for line in path.read_text().splitlines():
        hucs.extend(re.findall(r"(?<!\d)\d{8}(?!\d)", line))

    return list(dict.fromkeys(hucs))


def make_session():
    session = requests.Session()

    retries = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retries),
    )

    return session


def download_station(session, huc):
    output_file = OUTPUT_DIR / f"{huc}.csv"

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Skipping existing file: {huc}", flush=True)
        return "skipped", str(output_file)

    params = [
        ("huc", huc),
        ("mimeType", "csv"),
        ("providers", "NWIS"),
        ("providers", "STORET"),
        ("sorted", "no"),
    ]

    response = session.get(
        BASE_URL,
        params=params,
        timeout=(30, 300),
    )
    response.raise_for_status()

    content = response.content

    if not content.strip():
        raise ValueError("Empty response")

    # Reject HTML error pages instead of saving them as CSVs.
    first_line = content.lstrip()[:500].lower()
    if first_line.startswith(b"<!doctype html") or \
       first_line.startswith(b"<html"):
        raise ValueError("Received HTML instead of CSV")

    # Validate that the response is readable as a CSV.
    import io
    df = pd.read_csv(io.BytesIO(content), nrows=5)

    if df.empty and len(df.columns) == 0:
        raise ValueError("Response has no recognizable CSV columns")

    output_file.write_bytes(content)

    print(
        f"Downloaded {huc}: {output_file.stat().st_size:,} bytes",
        flush=True,
    )

    return "downloaded", str(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--huc-file",
        type=Path,
        default=DEFAULT_HUC_FILE,
    )
    parser.add_argument(
        "--n-hucs",
        type=int,
        default=10,
        help="Number of HUC8s to include in the pilot",
    )
    args = parser.parse_args()

    if not args.huc_file.exists():
        raise FileNotFoundError(
            f"HUC8 list not found: {args.huc_file}\n"
            "Pass its actual path using --huc-file."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hucs = read_hucs(args.huc_file)
    if not hucs:
        raise ValueError("No valid eight-digit HUC codes found")

    pilot_hucs = hucs[:args.n_hucs]

    print(f"Total HUC8s in list: {len(hucs)}", flush=True)
    print(f"HUC8s in this pilot: {len(pilot_hucs)}", flush=True)

    session = make_session()
    manifest = []

    for i, huc in enumerate(pilot_hucs, start=1):
        print(
            f"[{i}/{len(pilot_hucs)}] Processing HUC8 {huc}",
            flush=True,
        )

        try:
            status, path = download_station(session, huc)
            manifest.append({
                "huc8": huc,
                "status": status,
                "path": path,
                "error": "",
            })

        except Exception as exc:
            print(f"FAILED {huc}: {exc}", flush=True)
            manifest.append({
                "huc8": huc,
                "status": "failed",
                "path": "",
                "error": str(exc),
            })

        time.sleep(1)

    pd.DataFrame(manifest).to_csv(MANIFEST, index=False)

    print(f"Manifest saved to {MANIFEST}", flush=True)
    print(
        pd.DataFrame(manifest)["status"].value_counts().to_dict(),
        flush=True,
    )


if __name__ == "__main__":
    main()