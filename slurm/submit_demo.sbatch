#!/bin/bash
# Quick sanity-check job: runs alm.py's built-in demo (one realization,
# solves the ground state, checks force balance, saves a plot). Use this to
# confirm the env/partition setup works before submitting real jobs.
#
# Usage:
#   sbatch slurm/submit_demo.sbatch
#
#SBATCH --job-name=alm_demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

python3 alm.py
