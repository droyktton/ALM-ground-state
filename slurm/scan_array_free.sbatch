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
# Verified on frontend.fisica.cabib: `--gres=gpu:1` is accepted (gpu
# partition nodes carry mixed GRES types -- rtx3080/rtx3080ti/a10/rtx5070ti
# -- but the generic `gpu:1` request matches any of them) and `alm_env`
# already has a working cupy (14.0.1) that sees the device -- no separate
# cupy_env needed here, so common.sh's default env is used as-is.
#
# --gpu-mem-budget-mb default lowered to 400 (from run_structure_factor.py's
# own default of 1500): the gpu partition's OverSubscribe=YES:4 means several
# array tasks can land on the same physical GPU at once, and 1500MB/task
# OOM'd ~31% of a first 70-task run (mostly at large L) under that
# contention; 400MB/task re-ran all of them cleanly. Override with
# ALM_GPU_MEM_MB=<mb> if the partition is emptier and you want more headroom
# per task.
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
    --qmin 0.001 --qmax 0.15 -c 0.0 --bc "$bc" --gpu \
    --gpu-mem-budget-mb "${ALM_GPU_MEM_MB:-400}" --no-plot)
echo "$output"

result_line=$(echo "$output" | grep "^RESULT")

c=$(echo "$result_line"            | grep -oP ' c=\K[0-9.eE+-]+')
delta=$(echo "$result_line"        | grep -oP 'delta=\K[0-9.eE+-]+')
zeta_s=$(echo "$result_line"       | grep -oP 'zeta_s=\K[0-9.eE+-]+')
zeta_s_err=$(echo "$result_line"   | grep -oP 'zeta_s_err=\K[0-9.eE+-]+')
W2_direct=$(echo "$result_line"    | grep -oP 'W2_direct=\K[0-9.eE+-]+')
W2_parseval=$(echo "$result_line"  | grep -oP 'W2_parseval=\K[0-9.eE+-]+')

echo "$bc,$L,$n,$c,$delta,$NSAMPLES,$zeta_s,$zeta_s_err,$W2_direct,$W2_parseval" > "$result_file"

echo "wrote $result_file"
