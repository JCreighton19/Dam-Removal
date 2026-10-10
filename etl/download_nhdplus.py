import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

# Project paths and download settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent

huc4_file = PROJECT_ROOT / "data" / "processed" / "huc4_list.txt"
huc8_file = PROJECT_ROOT / "data" / "processed" / "huc8_list.txt"
output_dir = PROJECT_ROOT / "data" / "raw" / "nhdplus"
output_dir.mkdir(parents=True, exist_ok=True)

# USGS NHDPlus HR download directory
BASE_URL = "https://prd-tnm.s3.amazonaws.com/"
PREFIX = "StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/"

# Read unique, valid watershed codes. Keep them as strings to preserve
# leading zeros (e.g., "0101").
huc4s = {
    x.strip() for x in huc4_file.read_text().splitlines()
    if re.fullmatch(r"\d{4}", x.strip())
}
huc8s = {
    x.strip() for x in huc8_file.read_text().splitlines()
    if re.fullmatch(r"\d{8}", x.strip())
}


def list_s3_files():
    """Retrieve the available ZIP filenames and their S3 object keys."""
    files = []
    params = {"list-type": "2", "prefix": PREFIX}

    with requests.Session() as session:
        while True:
            response = session.get(BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            # Parse the S3 XML response using its default namespace.
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            files.extend(
                item.findtext("s3:Key", namespaces=ns)
                for item in root.findall("s3:Contents", ns)
                if item.findtext("s3:Key", namespaces=ns)
            )

            # Fetch additional pages if the directory listing is too large
            # to return in a single response.
            if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
                break

            token = root.findtext(
                "s3:NextContinuationToken", namespaces=ns
            )
            if not token:
                raise RuntimeError(
                    "S3 listing is truncated but has no continuation token."
                )

            params = {
                "list-type": "2",
                "prefix": PREFIX,
                "continuation-token": token,
            }

    # Map each ZIP filename to its full S3 key for downloading.
    return {
        key.rsplit("/", 1)[-1]: key
        for key in files
        if key.lower().endswith(".zip")
    }


def download(key):
    """Download a ZIP file, skipping existing nonempty files."""
    filename = key.rsplit("/", 1)[-1]
    output_file = output_dir / filename

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Already exists: {filename}")
        return

    # Use a temporary file so interrupted downloads aren't mistaken
    # for completed ZIP files.
    temp_file = output_file.with_suffix(output_file.suffix + ".part")
    url = BASE_URL + key

    print(f"Downloading {filename}...")

    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()

            # Stream in 1 MB chunks to limit memory usage.
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        temp_file.replace(output_file)
        print(f"Downloaded: {filename}")

    except Exception:
        temp_file.unlink(missing_ok=True)
        raise


# Discover the actual published files instead of assuming every HUC
# has a downloadable package.
available = list_s3_files()

# Index available packages by HUC code and resolution.
hu4_packages = {}
hu8_packages = {}

for filename, key in available.items():
    match = re.fullmatch(
        r"NHDPLUS_H_(\d{4}|\d{8})_HU(4|8)_GDB\.zip",
        filename,
        flags=re.IGNORECASE,
    )
    if not match:
        continue

    code, resolution = match.groups()

    if resolution == "4" and len(code) == 4:
        hu4_packages[code] = key
    elif resolution == "8" and len(code) == 8:
        hu8_packages[code] = key


# Prefer one HU4 package per requested HUC4. If unavailable, fall back
# to matching HU8 packages associated with our dam sites.
selected = []
missing = []

for huc4 in sorted(huc4s):
    if huc4 in hu4_packages:
        selected.append(hu4_packages[huc4])
        continue

    matching_huc8s = sorted(
        huc8 for huc8 in huc8s
        if huc8.startswith(huc4) and huc8 in hu8_packages
    )

    if matching_huc8s:
        selected.extend(hu8_packages[huc8] for huc8 in matching_huc8s)
    else:
        missing.append(huc4)

# Review coverage before downloading. A HUC4 may require multiple HU8
# packages, so the selected package count can exceed the HUC4 count.
print(f"Requested HUC4s: {len(huc4s)}")
print(f"Packages selected: {len(selected)}")
print(f"Missing HUC4s: {missing}")

for key in selected:
    download(key)

print("NHDPlus HR downloads complete.")