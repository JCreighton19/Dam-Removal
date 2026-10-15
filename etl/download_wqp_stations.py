from pathlib import Path
import argparse
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from etl.common.config import default_case_study, load_config

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


def download_station(session, huc, output_dir):
    output_file = output_dir / f"{huc}.csv"

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
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension). "
        "By default, downloads stations for exactly this case study's HUC8s.",
    )
    parser.add_argument(
        "--huc-file",
        type=Path,
        default=None,
        help="Override: read HUC8s from this file instead of the case-study "
        "config (e.g. to pull a broader exploratory sample of watersheds "
        "unrelated to any specific case study).",
    )
    parser.add_argument(
        "--n-hucs",
        type=int,
        default=None,
        help="Only used with --huc-file: cap the number of HUC8s read from it.",
    )
    args = parser.parse_args()

    if args.huc_file is not None:
        if not args.huc_file.exists():
            raise FileNotFoundError(
                f"HUC8 list not found: {args.huc_file}\n"
                "Pass its actual path using --huc-file."
            )
        hucs = read_hucs(args.huc_file)
        if not hucs:
            raise ValueError("No valid eight-digit HUC codes found")
        if args.n_hucs is not None:
            hucs = hucs[:args.n_hucs]
        project_root = Path(__file__).resolve().parents[1]
    else:
        cfg = load_config(args.case_study)
        hucs = cfg.huc8s
        project_root = cfg.project_root

    output_dir = project_root / "data" / "raw" / "wqp" / "stations"
    manifest_path = project_root / "data" / "raw" / "wqp" / "station_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"HUC8s to download: {len(hucs)}", flush=True)

    session = make_session()
    manifest = []

    for i, huc in enumerate(hucs, start=1):
        print(
            f"[{i}/{len(hucs)}] Processing HUC8 {huc}",
            flush=True,
        )

        try:
            status, path = download_station(session, huc, output_dir)
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

    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    print(f"Manifest saved to {manifest_path}", flush=True)
    print(
        pd.DataFrame(manifest)["status"].value_counts().to_dict(),
        flush=True,
    )


if __name__ == "__main__":
    main()