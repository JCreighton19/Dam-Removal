
#!/bin/bash
#SBATCH --job-name=dam_env
#SBATCH --output=data/slurm/logs/env_%j.out
#SBATCH --error=data/slurm/logs/env_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -e

cd /courses/DS5500.202710/teams/team4

module load anaconda3/2024.06

echo "Starting Conda environment setup: $(date)"

if conda env list | awk '{print $1}' | grep -qx "dam-removal"; then
    echo "Environment exists. Updating..."
    conda env update -n dam-removal -f environment.yml
else
    echo "Environment does not exist. Creating..."
    conda env create -f environment.yml
fi

echo "Environment setup completed: $(date)"