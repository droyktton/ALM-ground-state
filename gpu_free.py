"""
Batched GPU (cupy) implementation of the ALM free-boundary-condition
pipeline: disorder -> u -> structure factor.

Only bc="free" is supported here. alm.solve_ground_state_free has C
identically zero (no brentq root search -- see its docstring), so the whole
per-sample pipeline (cumsum -> elementwise power -> cumsum -> DCT) has no
data-dependent control flow and no CPU<->GPU sync points, which makes it
safe and effective to batch many disorder realizations into one set of 2D
array ops on the GPU. bc="periodic" still needs a scalar brentq root search
per sample and is not covered by this module.

Used by run_structure_factor.py's --gpu flag, and by
prototype_free_gpu.py's benchmark.
"""

import numpy as np

from alm import sample_disorder

try:
    import cupy as cp
    from cupyx.scipy.fft import dct as gpu_dct
    HAVE_CUPY = True
except Exception:
    cp = None
    gpu_dct = None
    HAVE_CUPY = False


def slope_from_stress_batched(xp, sigma, c, n, newton_iters=60, tol=1e-13):
    """Batched version of alm.slope_from_stress (any leading batch dims).

    Omits alm.py's per-element brentq safety net for non-converged Newton
    points (brentq has no vectorized/GPU form). Irrelevant for c=0 (no
    Newton at all) and negligible for moderate c/n in practice.
    """
    s0 = xp.sign(sigma) * xp.abs(sigma) ** (1.0 / (2 * n - 1))
    if c == 0.0:
        return s0
    s = s0
    for _ in range(newton_iters):
        resid = c * s + xp.sign(s) * xp.abs(s) ** (2 * n - 1) - sigma
        dresid = c + (2 * n - 1) * xp.abs(s) ** (2 * n - 2)
        step = resid / dresid
        s = s - step
        if xp.max(xp.abs(step)) < tol:
            break
    return s


def free_pipeline_batched(xp, f, c, n):
    """f: (batch, L) array of raw disorder samples. Returns u (batch, L),
    matching alm.solve_ground_state_free applied independently to each row.
    """
    f = f - f.mean(axis=1, keepdims=True)
    F_full = xp.cumsum(f, axis=1)
    sigma = -F_full[:, :-1]
    s = slope_from_stress_batched(xp, sigma, c, n)
    zeros_col = xp.zeros((f.shape[0], 1), dtype=s.dtype)
    u = xp.concatenate([zeros_col, xp.cumsum(s, axis=1)], axis=1)
    u = u - u.mean(axis=1, keepdims=True)
    return u


def structure_factor_sums_batched(xp, dct_fn, u, L):
    """Returns (S_sum, W2_sum) summed (not averaged) over the batch axis --
    caller accumulates across chunks and divides by n_samples at the end."""
    coef = dct_fn(u, type=2, norm="ortho", axis=-1)
    power = (coef ** 2) / L
    S_sum = power[:, 1:].sum(axis=0)        # drop k=0 mode
    W2_sum = xp.mean(u ** 2, axis=1).sum()
    return S_sum, W2_sum


def structure_factor_free_gpu(L, c, n, Delta, n_samples, rng, dist="gaussian",
                               mem_budget_bytes=1.5e9):
    """GPU entry point mirroring run_structure_factor.structure_factor_average's
    bc="free" branch. Draws disorder on CPU (via alm.sample_disorder, so the
    RNG consumption order/results match the CPU path exactly for a given
    seed), then processes it on the GPU in memory-bounded chunks over the
    sample axis -- needed because an unchunked (n_samples, L) batch can
    easily exceed GPU memory at large L.

    Returns (q, S_avg, W2_avg), same shapes/semantics as the CPU bc="free"
    path in run_structure_factor.py.
    """
    if not HAVE_CUPY:
        raise RuntimeError(
            "gpu_free.structure_factor_free_gpu requires cupy, which is not "
            "importable in this environment. Run in an environment with "
            "cupy installed (e.g. the 'cupy_env' conda env), or drop --gpu."
        )

    bytes_per_sample = 6 * 8 * L  # ~6 live (batch, L) float64 arrays through the pipeline
    chunk = max(1, int(mem_budget_bytes // bytes_per_sample))
    chunk = min(chunk, n_samples)

    S_accum = None
    W2_accum = 0.0
    remaining = n_samples
    while remaining > 0:
        this_chunk = min(chunk, remaining)
        f_chunk_cpu = np.stack(
            [sample_disorder(L, Delta, rng, dist=dist) for _ in range(this_chunk)]
        )
        f_chunk = cp.asarray(f_chunk_cpu)
        u = free_pipeline_batched(cp, f_chunk, c, n)
        S_sum, W2_sum = structure_factor_sums_batched(cp, gpu_dct, u, L)
        S_accum = S_sum if S_accum is None else (S_accum + S_sum)
        W2_accum += float(W2_sum)
        remaining -= this_chunk
        del f_chunk, u

    S_avg = cp.asnumpy(S_accum) / n_samples
    W2_avg = W2_accum / n_samples
    k = np.arange(1, L)
    q = np.pi * k / L
    return q, S_avg, W2_avg


def gpu_device_name():
    if not HAVE_CUPY:
        return None
    return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
