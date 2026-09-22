#!/bin/bash
# Array-job replacement for the bc=periodic half of scan_sweep_bc.sh: each
# array task runs exactly ONE (n, L) combination of run_structure_factor.py
# --bc periodic in parallel, on the CPU partition.
#
# bc=periodic needs a per-sample scalar brentq root search, which has no
# GPU implementation -- see gpu_free.py and slurm/scan_array_free.sbatch
# (the bc=free counterpart, which DOES run on GPU). A single sbatch script
# can't mix partitions/gres across array tasks, so periodic and free are
# submitted as two separate jobs; merge_scan_results.sh combines both
# jobs' output afterwards.
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
# Partition: cpu is the natural fit (no GPU path for periodic either way),
# but check `sinfo -p cpu` first -- if its nodes are drained/down, override
# at submit time instead of editing this file, e.g.:
#   sbatch --partition=gpu slurm/scan_array_periodic.sbatch
#
#SBATCH --job-name=alm_scan_periodic
#SBATCH --partition=cpu
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
bc=periodic

echo "task $SLURM_ARRAY_TASK_ID -> bc=$bc n=$n L=$L samples=$NSAMPLES"

mkdir -p slurm/results
result_file="slurm/results/result_${bc}_L${L}_n${n}.csv"

output=$(python3 run_structure_factor.py -L "$L" -n "$n" --samples "$NSAMPLES" \
    --qmin 0.015 --qmax 0.15 -c 0.0 --bc "$bc" --no-plot)
echo "$output"

result_line=$(echo "$output" | grep "^RESULT")

zeta_s=$(echo "$result_line"       | grep -oP 'zeta_s=\K[0-9.eE+-]+')
zeta_s_err=$(echo "$result_line"   | grep -oP 'zeta_s_err=\K[0-9.eE+-]+')
W2_direct=$(echo "$result_line"    | grep -oP 'W2_direct=\K[0-9.eE+-]+')
W2_parseval=$(echo "$result_line"  | grep -oP 'W2_parseval=\K[0-9.eE+-]+')

echo "$bc,$L,$n,1.0,1.0,$NSAMPLES,$zeta_s,$zeta_s_err,$W2_direct,$W2_parseval" > "$result_file"

echo "wrote $result_file"
