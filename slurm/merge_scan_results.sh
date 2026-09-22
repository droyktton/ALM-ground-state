#!/bin/bash
# Merge the per-task CSV rows written by scan_array_periodic.sbatch and
# scan_array_free.sbatch (one file per (bc, n, L) combination in
# slurm/results/) into a single scan_results.csv, in the same format
# plot_zeta_vs_n.py expects. Run this after all array tasks from BOTH jobs
# have finished (check with `squeue -u $USER` or `sacct`).
#
# Usage:
#   slurm/merge_scan_results.sh [output_csv]   # default: scan_results.csv
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

OUTFILE="${1:-scan_results.csv}"
RESULTS_DIR="slurm/results"

shopt -s nullglob
files=("$RESULTS_DIR"/result_*.csv)
if (( ${#files[@]} == 0 )); then
    echo "No result files found in $RESULTS_DIR/ -- have scan_array_periodic.sbatch / scan_array_free.sbatch finished?" >&2
    exit 1
fi

echo "bc,L,n,c,delta,samples,zeta_s,zeta_s_err,W2_direct,W2_parseval" > "$OUTFILE"
cat "${files[@]}" >> "$OUTFILE"

echo "Merged ${#files[@]} result files into $OUTFILE"
