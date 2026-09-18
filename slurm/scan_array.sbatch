#!/bin/bash
# Array-job version of scan_sweep_bc.sh: instead of looping over
# bc x n x L serially on one node, each array task runs exactly ONE
# (bc, n, L) combination of run_structure_factor.py in parallel.
#
# Grid below must match scan_sweep_bc.sh (BCS x NS x LS). Edit both the
# arrays and #SBATCH --array to match if you change the sweep.
#
# Usage:
#   sbatch slurm/scan_array.sbatch
#   # then, once all tasks finish:
#   slurm/merge_scan_results.sh
#
#SBATCH --job-name=alm_scan
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-119
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# ---- sweep grid: must match scan_sweep_bc.sh -------------------------
BCS=(periodic free)
NS=(0.8 0.85 0.9 1.0 1.1 1.3 1.5 2.0 3.0 4.0)
LS=(2048 8192 32768 131072 524288 1048576)
NSAMPLES=1000
# ------------------------------------------------------------------------

NBC=${#BCS[@]}
NN=${#NS[@]}
NL=${#LS[@]}
NTOTAL=$((NBC * NN * NL))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This script must be submitted as an array job (see --array above)." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= NTOTAL )); then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (grid has $NTOTAL combinations, 0-$((NTOTAL - 1)))." >&2
    exit 1
fi

# Decode task id -> (bc, n, L). Must stay consistent with merge_scan_results.sh
# only via the CSV files it reads, not this indexing, so it's safe to edit.
idx=$SLURM_ARRAY_TASK_ID
li=$(( idx % NL )); idx=$(( idx / NL ))
ni=$(( idx % NN )); idx=$(( idx / NN ))
bi=$(( idx % NBC ))

bc=${BCS[$bi]}
n=${NS[$ni]}
L=${LS[$li]}

echo "task $SLURM_ARRAY_TASK_ID -> bc=$bc n=$n L=$L samples=$NSAMPLES"

mkdir -p slurm/results
result_file="slurm/results/result_${bc}_L${L}_n${n}.csv"

output=$(python3 run_structure_factor.py -L "$L" -n "$n" --samples "$NSAMPLES" \
    --kmin 2 --kmax 100 -c 0.0 --bc "$bc" --no-plot)
echo "$output"

result_line=$(echo "$output" | grep "^RESULT")

zeta_s=$(echo "$result_line"       | grep -oP 'zeta_s=\K[0-9.eE+-]+')
zeta_s_err=$(echo "$result_line"   | grep -oP 'zeta_s_err=\K[0-9.eE+-]+')
W2_direct=$(echo "$result_line"    | grep -oP 'W2_direct=\K[0-9.eE+-]+')
W2_parseval=$(echo "$result_line"  | grep -oP 'W2_parseval=\K[0-9.eE+-]+')

echo "$bc,$L,$n,1.0,1.0,$NSAMPLES,$zeta_s,$zeta_s_err,$W2_direct,$W2_parseval" > "$result_file"

echo "wrote $result_file"
