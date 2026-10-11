from pathlib import Path
from io import StringIO
import argparse
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data/raw/wqp/results/pilot_station_selection.csv"
OUTPUT_DIR = ROOT / "data/raw/wqp/results/pilot_download"
MANIFEST_FILE = OUTPUT_DIR / "download_manifest.csv"

BASE_URL = "https://www.waterqualitydata.us/data/Result/search"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(
        INPUT_FILE,
        dtype={"MonitoringLocationIdentifier": "string"},
    )

    site_ids = (
        stations["MonitoringLocationIdentifier"]
        .dropna()
        .str.strip()
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .tolist()[:args.max_stations]
    )

    print(f"Stations selected: {len(site_ids)}", flush=True)

    session = requests.Session()

    retries = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    manifest = []

    for start in range(0, len(site_ids), args.batch_size):
        batch_num = start // args.batch_size + 1
        batch = site_ids[start:start + args.batch_size]

        output_file = OUTPUT_DIR / f"results_batch_{batch_num:03d}.csv"

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"Skipping existing {output_file.name}", flush=True)

            manifest.append({
                "batch": batch_num,
                "station_count": len(batch),
                "status": "already_exists",
                "file": output_file.name,
                "rows": None,
                "error": None,
            })
            continue

        print(
            f"Batch {batch_num}: requesting {len(batch)} stations",
            flush=True,
        )

        try:
            # Construct the query using one siteid parameter per station.
            params = [
                ("siteid", site_id)
                for site_id in batch
            ]

            params.extend([
                ("mimeType", "csv"),
                ("zip", "no"),
                ("sorted", "no"),
            ])

            response = session.get(
                BASE_URL,
                params=params,
                timeout=(30, 300),
            )

            print(f"  URL: {response.url}", flush=True)
            print(f"  HTTP status: {response.status_code}", flush=True)
            print(
                f"  Content-Type: "
                f"{response.headers.get('Content-Type')}",
                flush=True,
            )
            print(
                f"  Response size: {len(response.content):,} bytes",
                flush=True,
            )

            response.raise_for_status()

            if not response.text.strip():
                raise ValueError("WQP returned an empty response body.")

            print(f"  Response preview: {response.text[:300]!r}", flush=True)

            df = pd.read_csv(StringIO(response.text), low_memory=False)

            if "MonitoringLocationIdentifier" not in df.columns:
                raise ValueError(
                    f"Unexpected response columns: {df.columns.tolist()}"
                )

            row_count = len(df)

            if row_count == 0:
                print(
                    "  WARNING: CSV has headers but no measurement rows.",
                    flush=True,
                )

                status = "zero_rows"
            else:
                status = "success"

            df.to_csv(output_file, index=False)

            print(
                f"  Saved {row_count:,} rows to {output_file.name}",
                flush=True,
            )

            manifest.append({
                "batch": batch_num,
                "station_count": len(batch),
                "status": status,
                "file": output_file.name,
                "rows": row_count,
                "error": None,
            })

        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)

            manifest.append({
                "batch": batch_num,
                "station_count": len(batch),
                "status": "failed",
                "file": output_file.name,
                "rows": None,
                "error": str(exc),
            })

        pd.DataFrame(manifest).to_csv(MANIFEST_FILE, index=False)

        time.sleep(1)

    pd.DataFrame(manifest).to_csv(MANIFEST_FILE, index=False)

    print("\nDownload summary:", flush=True)
    print(
        pd.DataFrame(manifest)["status"].value_counts().to_string(),
        flush=True,
    )
    print(f"Manifest: {MANIFEST_FILE}", flush=True)


if __name__ == "__main__":
    main()