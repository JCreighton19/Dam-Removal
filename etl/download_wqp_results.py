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

Batches that return no results are recorded in the manifest but not
saved as CSVs. On a re-run, a batch is skipped only if the manifest shows
it was already downloaded (or came back empty) for the same stations and
date range, so changing wq_window or the station set re-requests it.
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

# Print a progress summary every this many batches.
PROGRESS_EVERY = 5


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

    print("=" * 70)
    print(f"Case study: {cfg.name}")
    print(f"HUC8s ({len(cfg.huc8s)}): {', '.join(cfg.huc8s)}")
    print(f"Window: {years_before} years before / {years_after} years after removal")
    print(
        f"Stations: {len(stations):,} in {len(batches):,} batches "
        f"of up to {args.batch_size}"
    )
    print(f"Output: {output_dir}")
    print("=" * 70, flush=True)

    warn_about_orphaned_batch_files(output_dir, len(batches))

    session = requests.Session()

    retries = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Keyed by batch number; earlier-run entries are kept until each batch
    # is revisited, so an interrupted run doesn't forget finished batches.
    manifest = {
        batch_num: entry
        for batch_num, entry in load_previous_manifest(manifest_file).items()
        if batch_num <= len(batches)
    }
    previous = dict(manifest)

    counts = {"data": 0, "empty": 0, "skipped": 0, "failed": 0}
    total_rows = 0
    started = time.monotonic()
    width = len(str(len(batches)))

    for batch_num, (start_year, end_year, batch) in enumerate(batches, start=1):
        output_file = output_dir / f"results_batch_{batch_num:03d}.csv"
        stations_key = ";".join(batch)
        prefix = (
            f"[{batch_num:>{width}}/{len(batches)}] "
            f"{start_year}-{end_year}, {len(batch)} stations:"
        )

        entry = {
            "batch": batch_num,
            "start_year": start_year,
            "end_year": end_year,
            "station_count": len(batch),
            "stations": stations_key,
            "status": None,
            "file": None,
            "rows": None,
            "error": None,
        }

        prior = previous.get(batch_num)

        if already_done(prior, start_year, end_year, stations_key, output_file):
            entry.update(status=prior["status"], file=prior["file"], rows=prior["rows"])
            counts["skipped"] += 1
            print(f"{prefix} skipped (done in an earlier run)", flush=True)
        else:
            try:
                df = request_batch(session, batch, start_year, end_year)

                if len(df):
                    df.to_csv(output_file, index=False)
                    entry.update(status="success", file=output_file.name, rows=len(df))
                    counts["data"] += 1
                    total_rows += len(df)
                    print(f"{prefix} {len(df):,} rows -> {output_file.name}", flush=True)
                else:
                    # Nothing to save; the manifest records that it was checked.
                    # Remove any file left by an earlier run with other stations.
                    output_file.unlink(missing_ok=True)
                    entry.update(status="zero_rows", rows=0)
                    counts["empty"] += 1
                    print(f"{prefix} no results", flush=True)

            except Exception as exc:
                entry.update(status="failed", error=str(exc))
                counts["failed"] += 1
                print(f"{prefix} FAILED: {exc}", flush=True)

            time.sleep(1)

        manifest[batch_num] = entry
        write_manifest(manifest, manifest_file)

        if batch_num % PROGRESS_EVERY == 0 or batch_num == len(batches):
            print_progress(batch_num, len(batches), counts, total_rows, started)

    failed = [b for b, e in sorted(manifest.items()) if e["status"] == "failed"]

    print("\n" + "=" * 70)
    print("Download finished")
    print(f"  Batches with data:  {counts['data']:,} ({total_rows:,} rows)")
    print(f"  Batches empty:      {counts['empty']:,} (not saved)")
    print(f"  Skipped (earlier):  {counts['skipped']:,}")
    print(f"  Failed:             {counts['failed']:,}")
    if failed:
        print(f"  Failed batch numbers: {failed} -- re-run to retry them")
    print(f"  Manifest: {manifest_file}")
    print("=" * 70, flush=True)


def request_batch(session, batch, start_year, end_year):
    """Request one batch of stations from WQP and return the results."""
    # One siteid parameter per station.
    params = [("siteid", site_id) for site_id in batch]
    params.extend([
        # WQP expects MM-DD-YYYY; both bounds are inclusive.
        ("startDateLo", f"01-01-{start_year}"),
        ("startDateHi", f"12-31-{end_year}"),
        ("mimeType", "csv"),
        ("zip", "no"),
        ("sorted", "no"),
    ])

    response = session.get(BASE_URL, params=params, timeout=(30, 300))
    response.raise_for_status()

    if not response.text.strip():
        raise ValueError(f"WQP returned an empty response body. URL: {response.url}")

    df = pd.read_csv(StringIO(response.text), low_memory=False)

    if "MonitoringLocationIdentifier" not in df.columns:
        raise ValueError(
            f"Unexpected response columns: {df.columns.tolist()[:10]}. "
            f"URL: {response.url}"
        )

    return df


def load_previous_manifest(manifest_file):
    """Earlier-run manifest entries keyed by batch number (empty if none)."""
    if not manifest_file.exists():
        return {}

    previous = pd.read_csv(manifest_file, dtype={"stations": str})
    previous = previous.astype(object).where(previous.notna(), None)
    return {int(row["batch"]): row for row in previous.to_dict("records")}


def already_done(prior, start_year, end_year, stations_key, output_file):
    """True if an earlier run finished this exact batch (same stations and
    date range) and, for batches with data, its CSV is still on disk."""
    if prior is None or prior.get("status") not in ("success", "zero_rows"):
        return False

    try:
        same_batch = (
            int(prior["start_year"]) == start_year
            and int(prior["end_year"]) == end_year
            and prior.get("stations") == stations_key
        )
    except (KeyError, TypeError, ValueError):
        return False  # manifest from an older version of this script

    if not same_batch:
        return False

    if prior["status"] == "success":
        return output_file.exists() and output_file.stat().st_size > 0

    return True


def write_manifest(manifest, manifest_file):
    pd.DataFrame([manifest[b] for b in sorted(manifest)]).to_csv(
        manifest_file, index=False
    )


def warn_about_orphaned_batch_files(output_dir, batch_count):
    orphaned = [
        path.name
        for path in sorted(output_dir.glob("results_batch_*.csv"))
        if int(path.stem.rsplit("_", 1)[-1]) > batch_count
    ]
    if orphaned:
        print(
            f"WARNING: {len(orphaned)} batch file(s) numbered above {batch_count} "
            f"are left over from an earlier run (e.g. {orphaned[0]}) and will be "
            "read by report_pilot_eligibility.py. Delete them if they're stale.",
            flush=True,
        )


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def print_progress(done, total, counts, total_rows, started):
    elapsed = time.monotonic() - started
    requested = done - counts["skipped"]
    # Estimate from requested batches only; skipped ones take no time.
    remaining = elapsed / requested * (total - done) if requested else 0

    print(
        f"--- {100 * done / total:5.1f}% ({done}/{total} batches) | "
        f"{total_rows:,} rows | {counts['data']} with data, "
        f"{counts['empty']} empty, {counts['failed']} failed, "
        f"{counts['skipped']} skipped | elapsed {format_duration(elapsed)}, "
        f"~{format_duration(remaining)} left ---",
        flush=True,
    )

if __name__ == "__main__":
    main()
