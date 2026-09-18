#!/bin/bash
# Sourced by the sbatch scripts in this folder: resolves the repo root (so
# jobs work no matter which directory `sbatch` was invoked from) and
# activates the conda environment used for this project.
#
# EDIT ME if your conda env name/location differs.

# Slurm copies the submitted batch script to a spool dir on the compute
# node before running it, so sibling files (this one included) can't be
# found relative to the running script's own path anymore. Use
# SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead --
# which is why every sbatch script here must be submitted from the repo
# root, e.g. `sbatch slurm/submit_demo.sbatch`.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"

# conda's own (de)activation hooks reference variables that don't exist yet
# (e.g. CONDA_BACKUP_CXX) -- harmless normally, but fatal under `set -u`
# (which the sbatch scripts sourcing this all set). Relax it just for conda.
set +u
source /home/koltona/miniconda3/etc/profile.d/conda.sh
conda activate alm_env
set -u

echo "[common.sh] host=$(hostname) project_dir=$PROJECT_DIR env=$(python3 -c 'import sys; print(sys.prefix)')"
