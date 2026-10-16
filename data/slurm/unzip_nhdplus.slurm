#!/bin/bash
#SBATCH --job-name=unzip_nhdplus
#SBATCH --output=data/slurm/logs/unzip_nhdplus_%j.out
#SBATCH --error=data/slurm/logs/unzip_nhdplus_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

ZIP_DIR="data/raw/nhdplus"
OUT_DIR="data/processed/nhdplus_extracted"

mkdir -p "$OUT_DIR"

echo "========================================"
echo "NHDPlus unzip job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Zips: $ZIP_DIR"
echo "Output: $OUT_DIR"
echo "Start time: $(date)"
echo "========================================"

shopt -s nullglob
zips=("$ZIP_DIR"/NHDPLUS_H_*_GDB.zip)

if [ ${#zips[@]} -eq 0 ]; then
    echo "No NHDPlus zips found in $ZIP_DIR -- run download_streams.slurm first."
    exit 1
fi

# Each zip extracts to $OUT_DIR/<zip stem>/, the layout
# etl/common/nhdplus.py:find_huc8_geodatabase expects.
for zip_path in "${zips[@]}"; do
    stem="$(basename "$zip_path" .zip)"
    target="$OUT_DIR/$stem"

    if [ -d "$target/$stem.gdb" ]; then
        echo "Skipping $stem (already extracted)"
        continue
    fi

    echo "Extracting $stem..."

    # Extract to a temporary directory first so a failed or partial
    # extraction is never mistaken for a finished one.
    tmp="$OUT_DIR/.tmp_$stem"
    rm -rf "$tmp"
    unzip -q -o "$zip_path" -d "$tmp"
    rm -rf "$target"
    mv "$tmp" "$target"

    if [ ! -d "$target/$stem.gdb" ]; then
        echo "WARNING: $stem.gdb not found directly under $target; contents:"
        ls "$target"
    fi
done

echo
echo "Extracted geodatabases:"
du -sh "$OUT_DIR"/*/

echo "NHDPlus unzip job finished: $(date)"
