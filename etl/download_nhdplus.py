import requests
from pathlib import Path

# Project root: the parent of the etl directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

huc4_file = PROJECT_ROOT / "data" / "processed" / "huc4_list.txt"
output_dir = PROJECT_ROOT / "data" / "raw" / "nhdplus"
output_dir.mkdir(parents=True, exist_ok=True)

huc4s = [
    x.strip()
    for x in huc4_file.read_text().splitlines()
    if x.strip()
]

for huc4 in huc4s:
    url = (
        f"https://prd-tnm.s3.amazonaws.com/"
        f"StagedProducts/Hydrography/NHDPlusHR/"
        f"VPU/HU4/GDB/NHDPLUS_H_{huc4}_HU4_GDB.zip"
    )

    output_file = output_dir / f"NHDPLUS_H_{huc4}_HU4_GDB.zip"

    if output_file.exists():
        print(f"Already exists: {huc4}")
        continue

    print(f"Downloading HUC4 {huc4}...")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print(f"Downloaded: {output_file}")