"""
Download WQP measurement results for stations confirmed relevant to a
dam removal by link_dams_stations.py.

Earlier versions of this script downloaded results for the full
candidate station list from select_candidate_stations.py (every
surface-water station in the case study's scope), capped by an arbitrary
--max-stations. That worked when the scope was one small watershed, but
once the scope spans multiple watersheds the candidate list can run into
the thousands while the stations actually network-connected to a dam are
a small fraction of that -- the cap was discarding almost all of the
relevant stations, essentially at random, instead of using the relevance
signal link_dams_stations.py already computed.

This now reads every data/processed/dam_station_links_<huc8>.csv for the
case study's HUC8 scope, keeps only rows classified as "upstream",
"downstream", or "same_reach" (excluding "unconnected"), and downloads
results for exactly that station set. It must run AFTER
link_dams_stations.py, not in parallel with it -- a HUC8 without a link
table yet (e.g. no NHDPlus coverage) is skipped with a warning rather
than treated as fatal, so a partially-processed case study still works.
"""

from pathlib import Path
from io import StringIO
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from etl.common.config import default_case_study, load_config
from etl.common.links import connected_only, load_links

BASE_URL = "https://www.waterqualitydata.us/data/Result/search"


def load_confirmed_stations(cfg):
    """Station IDs confirmed network-connected to a dam, across every
    HUC8 this case study has a link table for."""
    processed_dir = cfg.project_root / "data" / "processed"
    station_ids = []
    missing_huc8s = []

    for huc8 in cfg.huc8s:
        links_path = processed_dir / f"dam_station_links_{huc8}.csv"

        if not links_path.exists():
            missing_huc8s.append(huc8)
            continue

        links = load_links(links_path)
        connected = connected_only(links)
        station_ids.extend(
            connected["station_id"].dropna().astype(str).str.strip().tolist()
        )

    if missing_huc8s:
        print(
            f"WARNING: no dam_station_links_*.csv yet for HUC8(s) "
            f"{missing_huc8s} -- run link_dams_stations.py for them first "
            "if they should be included here.",
            flush=True,
        )

    return sorted({sid for sid in station_ids if sid})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Optional cap for testing; defaults to every confirmed "
        "network-connected station.",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.case_study)

    output_dir = cfg.project_root / "data" / "raw" / "wqp" / "results" / "pilot_download"
    manifest_file = output_dir / "download_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    site_ids = load_confirmed_stations(cfg)

    if not site_ids:
        raise ValueError(
            "No network-connected stations found across this case study's "
            "HUC8s. Run link_dams_stations.py first."
        )

    if args.max_stations is not None:
        site_ids = site_ids[:args.max_stations]

    print(f"Confirmed network-connected stations: {len(site_ids)}", flush=True)

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

        output_file = output_dir / f"results_batch_{batch_num:03d}.csv"

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

        pd.DataFrame(manifest).to_csv(manifest_file, index=False)

        time.sleep(1)

    pd.DataFrame(manifest).to_csv(manifest_file, index=False)

    print("\nDownload summary:", flush=True)
    print(
        pd.DataFrame(manifest)["status"].value_counts().to_string(),
        flush=True,
    )
    print(f"Manifest: {manifest_file}", flush=True)


if __name__ == "__main__":
    main()
