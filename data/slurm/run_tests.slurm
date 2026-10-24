#!/bin/bash
#SBATCH --job-name=tests
#SBATCH --output=data/slurm/logs/tests_%j.out
#SBATCH --error=data/slurm/logs/tests_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Runs every tests/test_*.py. Passing tests print "ok"; a failing test's
# output goes to the .err file. Exits non-zero if any test fails.

set -uo pipefail

cd /courses/DS5500.202710/teams/team4/

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dam-removal

echo "Job started: $(date)"

failed=0
for t in tests/test_*.py; do
    if python "$t" > /dev/null; then
        echo "ok   $t"
    else
        echo "FAIL $t"
        failed=1
    fi
done

echo "Job completed: $(date)"
exit $failed
