#!/bin/bash
# Sweep over several n values, L values, and BOTH boundary conditions,
# extract zeta_s and W^2(L), and collect results into a CSV.
#
# Set ALM_GPU=1 to run the bc=free half on GPU (via run_structure_factor.py
# --gpu). Off by default since it needs cupy (e.g. `conda activate
# cupy_env`) and bc=periodic never uses it anyway.
#   ALM_GPU=1 ./scan_sweep_bc.sh

GPU_FLAG=""
if [[ "${ALM_GPU:-0}" == "1" ]]; then
    GPU_FLAG="--gpu"
fi

OUTFILE="scan_results.csv"
echo "bc,L,n,c,delta,samples,zeta_s,zeta_s_err,W2_direct,W2_parseval" > "$OUTFILE"

for bc in periodic free; do
    for n in 0.8 0.85 0.9 1.0 1.1 1.3 1.5 2.0 3.0 4.0; do
        for L in 2048 8192 32768 131072 524288 1048576 2097152; do
            #nsamples=$(echo "1024*32768/$L" | bc)  # Number of samples to average over for each (L,n) pair
            nsamples=100

            bc_gpu_flag=""
            if [[ "$bc" == "free" ]]; then
                bc_gpu_flag="$GPU_FLAG"
            fi

            # Run the tool, capture full stdout, skip plotting/showing to keep it fast.
            # Fixed physical q-window (same qmin/qmax for every L), so the fit
            # avoids both the small-q boundary/finite-size region and the
            # large-q small-scale/lattice crossover, independent of L.
            output=$(python3 run_structure_factor.py -L "$L" -n "$n" --samples $nsamples --qmin 0.015 --qmax 0.15 -c 0.0 --bc $bc $bc_gpu_flag --no-plot)

            # Grab the single machine-readable line and turn it into a CSV row.
            # The RESULT line looks like:
            #   RESULT L=4096 n=2.0 c=1.0 delta=1.0 samples=200 dist=gaussian bc=periodic zeta_s=1.4164 ...
            result_line=$(echo "$output" | grep "^RESULT")

            zeta_s=$(echo "$result_line"   | grep -oP 'zeta_s=\K[0-9.eE+-]+')
            zeta_s_err=$(echo "$result_line" | grep -oP 'zeta_s_err=\K[0-9.eE+-]+')
            W2_direct=$(echo "$result_line"  | grep -oP 'W2_direct=\K[0-9.eE+-]+')
            W2_parseval=$(echo "$result_line" | grep -oP 'W2_parseval=\K[0-9.eE+-]+')

            echo "$bc,$L,$n,1.0,1.0,$nsamples,$zeta_s,$zeta_s_err,$W2_direct,$W2_parseval" >> "$OUTFILE"

            echo "bc=$bc L=$L samples=$nsamples n=$n -> zeta_s=$zeta_s +/- $zeta_s_err   W2=$W2_direct"
        done
    done
done

echo "Saved sweep results to $OUTFILE"
