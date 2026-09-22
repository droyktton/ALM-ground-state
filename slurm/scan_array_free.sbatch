#!/bin/bash
# Array-job replacement for the bc=free half of scan_sweep_bc.sh: each
# array task runs exactly ONE (n, L) combination of run_structure_factor.py
# --bc free --gpu in parallel, on the GPU partition.
#
# bc=free has no root search (C=0 exactly -- see gpu_free.py), so unlike
# bc=periodic it batches cleanly onto a GPU. See
# slurm/scan_array_periodic.sbatch for the CPU/periodic counterpart; run
# both jobs and merge_scan_results.sh combines their output.
#
# ***UNVERIFIED ON THE CLUSTER*** -- this was written from a workstation,
# not frontend.fisica.cabib, so the following need checking before/at first
# submission:
#   - `--gres=gpu:1` below: confirm the actual GRES name with
#     `scontrol show partition gpu` (some clusters use gpu:a100:1 etc.).
#   - The `cupy_env` conda env must exist on the cluster (it only exists on
#     this workstation right now) with a cupy build matching the GPU
#     node's CUDA driver -- check via `nvidia-smi` on a gpu-partition node,
#     then `pip install cupy-cudaXXx` to match. See slurm/setup_env.sh for
#     the analogous one-off CPU-env setup pattern.
#
# Grid below must match scan_sweep_bc.sh (NS x LS). Edit both the arrays
# and #SBATCH --array to match if you change the sweep.
#
# Usage:
#   sbatch slurm/scan_array_periodic.sbatch
#   sbatch slurm/scan_array_free.sbatch
#   # then, once both have finished:
#   slurm/merge_scan_results.sh
#
#SBATCH --job-name=alm_scan_free
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-69
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail
export ALM_CONDA_ENV=cupy_env
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# ---- sweep grid: must match scan_sweep_bc.sh --------------------------
NS=(0.8 0.85 0.9 1.0 1.1 1.3 1.5 2.0 3.0 4.0)
LS=(2048 8192 32768 131072 524288 1048576 2097152)
NSAMPLES=1000
# ------------------------------------------------------------------------

NN=${#NS[@]}
NL=${#LS[@]}
NTOTAL=$((NN * NL))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This script must be submitted as an array job (see --array above)." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= NTOTAL )); then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (grid has $NTOTAL combinations, 0-$((NTOTAL - 1)))." >&2
    exit 1
fi

# Decode task id -> (n, L). Must stay consistent with merge_scan_results.sh
# only via the CSV files it reads, not this indexing, so it's safe to edit.
idx=$SLURM_ARRAY_TASK_ID
li=$(( idx % NL )); idx=$(( idx / NL ))
ni=$(( idx % NN ))

n=${NS[$ni]}
L=${LS[$li]}
bc=free

echo "task $SLURM_ARRAY_TASK_ID -> bc=$bc n=$n L=$L samples=$NSAMPLES"

mkdir -p slurm/results
result_file="slurm/results/result_${bc}_L${L}_n${n}.csv"

output=$(python3 run_structure_factor.py -L "$L" -n "$n" --samples "$NSAMPLES" \
    --kmin 2 --kmax 100 -c 0.0 --bc "$bc" --gpu --no-plot)
echo "$output"

result_line=$(echo "$output" | grep "^RESULT")

zeta_s=$(echo "$result_line"       | grep -oP 'zeta_s=\K[0-9.eE+-]+')
zeta_s_err=$(echo "$result_line"   | grep -oP 'zeta_s_err=\K[0-9.eE+-]+')
W2_direct=$(echo "$result_line"    | grep -oP 'W2_direct=\K[0-9.eE+-]+')
W2_parseval=$(echo "$result_line"  | grep -oP 'W2_parseval=\K[0-9.eE+-]+')

echo "$bc,$L,$n,1.0,1.0,$NSAMPLES,$zeta_s,$zeta_s_err,$W2_direct,$W2_parseval" > "$result_file"

echo "wrote $result_file"
