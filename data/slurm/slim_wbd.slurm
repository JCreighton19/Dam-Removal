#!/bin/bash
#SBATCH --job-name=slim_wbd
#SBATCH --output=data/slurm/logs/slim_wbd_%j.out
#SBATCH --error=data/slurm/logs/slim_wbd_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

source data/slurm/case_study.env

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Shrinking the national WBD to its HUC8 layer (one-time; unzip_huc8 also runs this)"

python -u etl/slim_wbd.py

ls -la data/raw/wbd/

echo "Job completed: $(date)"
