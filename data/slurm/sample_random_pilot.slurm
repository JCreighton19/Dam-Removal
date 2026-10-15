#!/bin/bash
#SBATCH --job-name=sample_random_pilot
#SBATCH --output=data/slurm/logs/sample_random_pilot_%j.out
#SBATCH --error=data/slurm/logs/sample_random_pilot_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -e

cd /courses/DS5500.202710/teams/team4

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Starting random pilot sampling"
echo "Start time: $(date)"

# Edit these as needed, or just run the script directly in an interactive
# session -- this is a one-off setup task, not a recurring pipeline stage.
python -u etl/sample_random_pilot.py --n-dams 5 --seed 42

echo "Finished: $(date)"
