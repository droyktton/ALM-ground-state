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


# numpy/scipy's BLAS/FFT backends default to using every core on the node
# unless told otherwise. Array-job tasks request --cpus-per-task=1 but run
# many-at-a-time on the same node, so without this every task tries to grab
# all cores and they all slow each other down (seen firsthand: 10 concurrent
# `periodic` tasks at L=2097152 blew past a 2h time limit that a single,
# unshared run finished in ~39 min). Pin every task to the 1 core it asked for.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[common.sh] host=$(hostname) project_dir=$PROJECT_DIR env=$(python3 -c 'import sys; print(sys.prefix)')"
