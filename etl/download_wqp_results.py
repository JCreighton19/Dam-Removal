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

Requests are also limited to the case study's `wq_window`: each station
is queried only for [earliest removal year - years_before, latest removal
year + years_after] across the dams it is connected to, matching the
windows report_pilot_eligibility.py evaluates. Stations whose connected
dams all lack a removal year are skipped, since the eligibility report
excludes them anyway. Stations sharing a date range are batched together,
because WQP applies one date range per request.

Existing batch files are skipped by batch number, so delete
data/raw/wqp/results/pilot_download/ after changing wq_window or the
station set.
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


def load_confirmed_stations(cfg, years_before, years_after):
    """Stations confirmed network-connected to a dam, across every HUC8
    this case study has a link table for, with the year range to request
    for each (start_year, end_year, inclusive)."""
    processed_dir = cfg.project_root / "data" / "processed"
    frames = []
    missing_huc8s = []

    for huc8 in cfg.huc8s:
        links_path = processed_dir / f"dam_station_links_{huc8}.csv"

        if not links_path.exists():
            missing_huc8s.append(huc8)
            continue

        connected = connected_only(load_links(links_path))
        frames.append(connected[["station_id", "year_removed"]])

    if missing_huc8s:
        print(
            f"WARNING: no dam_station_links_*.csv yet for HUC8(s) "
            f"{missing_huc8s} -- run link_dams_stations.py for them first "
            "if they should be included here.",
            flush=True,
        )

    columns = ["station_id", "start_year", "end_year"]
    if not frames:
        return pd.DataFrame(columns=columns)

    pairs = pd.concat(frames, ignore_index=True).dropna(subset=["station_id"])
    pairs["station_id"] = pairs["station_id"].astype(str).str.strip()
    pairs = pairs[pairs["station_id"].ne("")]
    pairs["year_removed"] = pd.to_numeric(pairs["year_removed"], errors="coerce")

    # Same rule as report_pilot_eligibility.prepare_links: missing or zero
    # removal years can't be placed in a window.
    connected_count = pairs["station_id"].nunique()
    pairs = pairs[pairs["year_removed"].notna() & pairs["year_removed"].ne(0)]
    no_year_count = connected_count - pairs["station_id"].nunique()

    if no_year_count:
        print(
            f"Skipping {no_year_count} connected station(s) whose dams all "
            "lack a removal year.",
            flush=True,
        )

    stations = (
        pairs.groupby("station_id")["year_removed"]
        .agg(["min", "max"])
        .reset_index()
    )
    stations["start_year"] = (stations["min"] - years_before).astype(int)
    stations["end_year"] = (stations["max"] + years_after).astype(int)

    return stations[columns].sort_values(columns).reset_index(drop=True)


def build_batches(stations, batch_size):
    """Group stations sharing a date range, then split into batches of
    (start_year, end_year, [station_id, ...])."""
    batches = []

    for (start_year, end_year), group in stations.groupby(
        ["start_year", "end_year"], sort=True
    ):
        ids = group["station_id"].tolist()
        for i in range(0, len(ids), batch_size):
            batches.append((int(start_year), int(end_year), ids[i:i + batch_size]))

    return batches


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

    wq_window = cfg.get("wq_window", {}) or {}
    years_before = int(wq_window.get("years_before", 3))
    years_after = int(wq_window.get("years_after", 3))

    stations = load_confirmed_stations(cfg, years_before, years_after)

    if stations.empty:
        raise ValueError(
            "No network-connected stations with a dam removal year found "
            "across this case study's HUC8s. Run link_dams_stations.py first."
        )

    if args.max_stations is not None:
        stations = stations.head(args.max_stations)

    batches = build_batches(stations, args.batch_size)

    print("HUC8s in case study:", len(cfg.huc8s))
    print("HUC8s:", cfg.huc8s)
    print(f"Window: {years_before} years before, {years_after} years after removal")
    print("Unique confirmed station IDs:", len(stations))
    print(f"Batches at batch size {args.batch_size}:", len(batches))
    print(f"Confirmed network-connected stations: {len(stations)}", flush=True)

    session = requests.Session()

    retries = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    manifest = []

    for batch_num, (start_year, end_year, batch) in enumerate(batches, start=1):
        output_file = output_dir / f"results_batch_{batch_num:03d}.csv"

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"Skipping existing {output_file.name}", flush=True)

            manifest.append({
                "batch": batch_num,
                "station_count": len(batch),
                "start_year": start_year,
                "end_year": end_year,
                "status": "already_exists",
                "file": output_file.name,
                "rows": None,
                "error": None,
            })
            continue

        print(
            f"Batch {batch_num}: requesting {len(batch)} stations "
            f"({start_year}-{end_year})",
            flush=True,
        )

        try:
            # Construct the query using one siteid parameter per station.
            params = [
                ("siteid", site_id)
                for site_id in batch
            ]

            params.extend([
                # WQP expects MM-DD-YYYY; both bounds are inclusive.
                ("startDateLo", f"01-01-{start_year}"),
                ("startDateHi", f"12-31-{end_year}"),
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
                "start_year": start_year,
                "end_year": end_year,
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
                "start_year": start_year,
                "end_year": end_year,
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
