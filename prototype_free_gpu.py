"""
Benchmark: gpu_free.structure_factor_free_gpu (used by run_structure_factor.py's
--gpu flag) against the current per-sample numpy implementation, bc="free".

Why free BC only
-----------------
alm.solve_ground_state_free has C identically zero (no brentq root search --
see its docstring), so the whole per-sample pipeline

    f -> cumsum -> sigma=-F[:-1] -> slope_from_stress -> cumsum -> u -> dct(u)

is pure elementwise/cumsum/FFT work with no data-dependent control flow and
no CPU<->GPU sync points, which is what makes it batchable/GPU-friendly (see
gpu_free.py for the batched implementation itself).

This script checks correctness against the existing scalar implementation
first, then benchmarks three variants:
  1. baseline    -- current code path: a Python loop calling
                    alm.solve_ground_state_free once per sample
                    (scipy.fft.dct per sample too).
  2. batched-cpu -- same math, batched across samples with numpy + scipy's
                    batched dct(axis=-1). Isolates the benefit of batching
                    alone, no GPU involved.
  3. batched-gpu -- gpu_free.structure_factor_free_gpu (cupy + cupyx.scipy.fft.dct),
                    processed in memory-bounded chunks.

Run inside the `cupy_env` conda environment (has cupy 14.1.1 + a visible
GPU). Falls back to CPU-only (skips variant 3) if cupy/a GPU is unavailable.
"""

import argparse
import time

import numpy as np
from scipy.fft import dct as cpu_dct

from alm import sample_disorder, solve_ground_state_free
from gpu_free import (
    HAVE_CUPY,
    free_pipeline_batched,
    structure_factor_sums_batched,
    structure_factor_free_gpu,
    gpu_device_name,
)

if HAVE_CUPY:
    import cupy as cp


def free_batch_cpu_chunked(f_full, c, n, L, mem_budget_bytes=1.5e9):
    """CPU-batched counterpart of gpu_free.structure_factor_free_gpu, chunked
    over the sample axis to bound host RAM -- this box only has 15 GB total,
    and an unchunked (n_samples, L) batch at L~2e6 blows past what's
    available once numpy's elementwise temporaries are accounted for."""
    n_samples = f_full.shape[0]
    bytes_per_sample = 6 * 8 * L
    chunk = max(1, int(mem_budget_bytes // bytes_per_sample))
    chunk = min(chunk, n_samples)

    S_accum = None
    W2_accum = 0.0
    for start in range(0, n_samples, chunk):
        f_chunk = f_full[start:start + chunk]
        u = free_pipeline_batched(np, f_chunk, c, n)
        S_sum, W2_sum = structure_factor_sums_batched(np, cpu_dct, u, L)
        S_accum = S_sum if S_accum is None else (S_accum + S_sum)
        W2_accum += float(W2_sum)
    return S_accum / n_samples, W2_accum / n_samples, chunk


def baseline_loop(f_full, c, n, L):
    """Current implementation's math path (mirrors
    run_structure_factor.structure_factor_average, bc='free'), but reusing
    pre-drawn disorder so all three variants see identical input."""
    n_samples = f_full.shape[0]
    S_accum = np.zeros(L - 1)
    W2_accum = 0.0
    for i in range(n_samples):
        u, s, F_full = solve_ground_state_free(f_full[i], c, n)
        coef = cpu_dct(u, type=2, norm="ortho")
        power = (coef ** 2) / L
        S_accum += power[1:]
        W2_accum += np.mean(u ** 2)
    return S_accum / n_samples, W2_accum / n_samples


# ----------------------------------------------------------------------
# Correctness check
# ----------------------------------------------------------------------
def check_correctness(L=4096, n_samples=24, c=0.0, n=2.0, seed=0):
    rng = np.random.default_rng(seed)
    f_full = np.stack([sample_disorder(L, 1.0, rng) for _ in range(n_samples)])

    S_base, W2_base = baseline_loop(f_full, c, n, L)
    S_cpu, W2_cpu, _ = free_batch_cpu_chunked(f_full, c, n, L)

    err_S_cpu = np.max(np.abs(S_cpu - S_base)) / np.max(np.abs(S_base))
    err_W2_cpu = abs(W2_cpu - W2_base) / abs(W2_base)
    print(f"[correctness] batched-cpu  vs baseline: max relerr S(q)={err_S_cpu:.2e}  W2={err_W2_cpu:.2e}")
    assert err_S_cpu < 1e-9 and err_W2_cpu < 1e-9, "batched-cpu diverges from baseline"

    if HAVE_CUPY:
        rng_gpu = np.random.default_rng(seed)  # same seed -> identical draws to f_full
        _, S_gpu, W2_gpu = structure_factor_free_gpu(L, c, n, 1.0, n_samples, rng_gpu)
        err_S_gpu = np.max(np.abs(S_gpu - S_base)) / np.max(np.abs(S_base))
        err_W2_gpu = abs(W2_gpu - W2_base) / abs(W2_base)
        print(f"[correctness] batched-gpu  vs baseline: max relerr S(q)={err_S_gpu:.2e}  W2={err_W2_gpu:.2e}")
        assert err_S_gpu < 1e-6 and err_W2_gpu < 1e-6, "batched-gpu diverges from baseline"
    print("[correctness] OK\n")


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
def benchmark(L_values, n_samples, c=0.0, n=2.0, seed=0, mem_budget_bytes=1.5e9):
    header = f"{'L':>10} {'n_samples':>9} {'baseline(s)':>12} {'batched-cpu(s)':>15} {'batched-gpu(s)':>15} {'cpu speedup':>12} {'gpu speedup':>12}"
    print(header)
    print("-" * len(header))
    for L in L_values:
        rng = np.random.default_rng(seed)
        f_full = np.stack([sample_disorder(L, 1.0, rng) for _ in range(n_samples)])

        t0 = time.perf_counter()
        baseline_loop(f_full, c, n, L)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        free_batch_cpu_chunked(f_full, c, n, L, mem_budget_bytes)
        t_cpu = time.perf_counter() - t0

        if HAVE_CUPY:
            # warm up cupy/cuFFT plan cache so the timed run isn't paying
            # for one-time JIT/plan compilation
            structure_factor_free_gpu(L, c, n, 1.0, min(4, n_samples),
                                       np.random.default_rng(seed), mem_budget_bytes=mem_budget_bytes)
            t0 = time.perf_counter()
            structure_factor_free_gpu(L, c, n, 1.0, n_samples,
                                       np.random.default_rng(seed), mem_budget_bytes=mem_budget_bytes)
            t_gpu = time.perf_counter() - t0
            gpu_str = f"{t_gpu:15.4f}"
            speedup_gpu = f"{t_base / t_gpu:12.2f}"
        else:
            gpu_str = f"{'n/a':>15}"
            speedup_gpu = f"{'n/a':>12}"

        print(f"{L:10d} {n_samples:9d} {t_base:12.4f} {t_cpu:15.4f} {gpu_str} "
              f"{t_base / t_cpu:12.2f} {speedup_gpu}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--Ls", type=int, nargs="+",
                         default=[16384, 65536, 262144, 1048576, 2097152])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("-n", type=float, default=2.0)
    parser.add_argument("-c", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mem-budget-mb", type=float, default=1500)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    print(f"cupy available: {HAVE_CUPY}")
    if HAVE_CUPY:
        dev = cp.cuda.Device(0)
        print(f"GPU: {gpu_device_name()}  "
              f"free/total mem: {dev.mem_info[0]/1e9:.2f}/{dev.mem_info[1]/1e9:.2f} GB\n")

    if not args.skip_correctness:
        check_correctness(c=args.c, n=args.n)

    benchmark(args.Ls, args.samples, c=args.c, n=args.n, seed=args.seed,
              mem_budget_bytes=args.mem_budget_mb * 1e6)


if __name__ == "__main__":
    main()
