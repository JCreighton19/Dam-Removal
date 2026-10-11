#!/bin/bash
#SBATCH --job-name=link_dams
#SBATCH --output=data/slurm/logs/link_dams_%j.out
#SBATCH --error=data/slurm/logs/link_dams_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd /courses/DS5500.202710/teams/team4/

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"
echo "Running dam-station relationship analysis"

python -u etl/link_dams_stations.py --huc8 19020401

echo "Job completed: $(date)"
