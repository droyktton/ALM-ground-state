#!/bin/bash
# Run a single run_structure_factor.py job on the cluster. All arguments
# given to `sbatch` after the script path are forwarded verbatim to
# run_structure_factor.py.
#
# Usage:
#   sbatch slurm/submit_single.sbatch -L 65536 -n 2.0 -c 1.0 --samples 500 \
#       --bc periodic --no-plot --no-show -o slurm/results/sf_L65536_n2.0.png
#
# Adjust --time/--mem below for large L / many samples.
#
#SBATCH --job-name=alm_sf
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

python3 run_structure_factor.py "$@"
