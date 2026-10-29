#!/bin/bash
#SBATCH --job-name=export_analysis_csv
#SBATCH --output=data/slurm/logs/export_analysis_csv_%j.out
#SBATCH --error=data/slurm/logs/export_analysis_csv_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Writes analysis_measurement.parquet to a CSV for manual inspection
# (read-only on the warehouse; run after build_analysis_dataset).

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Case study: $DAM_REMOVAL_CASE_STUDY"

SOURCE="data/warehouse/${DAM_REMOVAL_CASE_STUDY}/analysis_measurement.parquet"
OUTPUT="data/scratch/analysis_measurement_${DAM_REMOVAL_CASE_STUDY}.csv"
mkdir -p data/scratch

echo "Exporting $SOURCE -> $OUTPUT"

python -u - "$SOURCE" "$OUTPUT" <<'EOF'
import sys
import duckdb

source, output = sys.argv[1], sys.argv[2]
duckdb.sql(f"""
    COPY (SELECT * FROM read_parquet('{source}')
          ORDER BY AR_ID, station_id, parameter_std, sample_date)
    TO '{output}' (HEADER)
""")
rows = duckdb.sql(f"SELECT count(*) FROM read_parquet('{source}')").fetchone()[0]
print(f"Rows written: {rows:,}")
EOF

ls -lh "$OUTPUT"

echo "Job completed: $(date)"
