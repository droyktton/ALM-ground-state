"""
Anharmonic Larkin Model (ALM) -- exact ground state solver
============================================================

Hamiltonian:
    H[u] = sum_i [ (c/2) s_i^2 + (1/2n) |s_i|^2n - f_i u_i ],   s_i = u_{i+1} - u_i

Force balance:
    sigma_i - sigma_{i-1} = f_i,   sigma(s) = c*s + |s|^(2n-2) s

This reduces the L-dimensional minimization to a single scalar root-finding
problem for the integration constant C, exactly as derived in the model
writeup. This script implements the construction end-to-end.
"""

import numpy as np
from scipy.optimize import brentq


def sample_disorder(L, Delta, rng, dist="gaussian"):
    """
    Draw L i.i.d. quenched-force samples with variance Delta, from one of
    several distributions -- all matched to the same variance, so the only
    thing that changes is the shape of the microscopic distribution (its
    higher moments / tail behavior), not its scale.

    dist options:
      "gaussian" : f ~ N(0, Delta)                          (kurtosis 3, unbounded tails)
      "uniform"  : f ~ Uniform(-a, a), a = sqrt(3*Delta)     (kurtosis 1.8, bounded)
      "bimodal"  : f = +/- sqrt(Delta) with equal probability (kurtosis 1, most non-Gaussian
                    finite-variance case possible -- two delta-function peaks)

    Zero-mean enforcement (sum_i f_i = 0) is handled separately by
    solve_ground_state; this function just draws the raw i.i.d. samples.
    """
    if dist == "gaussian":
        return rng.normal(0.0, np.sqrt(Delta), size=L)
    elif dist == "uniform":
        a = np.sqrt(3.0 * Delta)
        return rng.uniform(-a, a, size=L)
    elif dist == "bimodal":
        sign = rng.choice([-1.0, 1.0], size=L)
        return sign * np.sqrt(Delta)
    else:
        raise ValueError(f"Unknown dist '{dist}'. Choose from: gaussian, uniform, bimodal.")


# ----------------------------------------------------------------------
# Constitutive relation:  sigma = c*s + sign(s)|s|^(2n-1)   -->  invert for s
# ----------------------------------------------------------------------
def slope_from_stress(sigma, c, n, newton_iters=60, tol=1e-13):
    """
    Solve  c*s + sign(s)|s|^(2n-1) = sigma  for s, elementwise, vectorized.

    For c == 0 this is explicit: s = sign(sigma) |sigma|^{1/(2n-1)}.
    For c > 0 the map s -> sigma(s) is strictly increasing (linear term +
    increasing power term), so each scalar equation has a unique root.
    We solve all sites simultaneously with a safeguarded Newton iteration,
    initialized from the exact c=0 solution (an excellent starting guess
    whenever the anharmonic term is not negligible, and exact when c=0).
    """
    sigma = np.asarray(sigma, dtype=float)
    s0 = np.sign(sigma) * np.abs(sigma) ** (1.0 / (2 * n - 1))

    if c == 0.0:
        return s0

    s = s0.copy()
    for _ in range(newton_iters):
        resid = c * s + np.sign(s) * np.abs(s) ** (2 * n - 1) - sigma
        dresid = c + (2 * n - 1) * np.abs(s) ** (2 * n - 2)
        step = resid / dresid
        s_new = s - step
        if np.max(np.abs(step)) < tol:
            s = s_new
            break
        s = s_new

    # safety net: polish any poorly converged points with bisection
    resid = c * s + np.sign(s) * np.abs(s) ** (2 * n - 1) - sigma
    bad = np.abs(resid) > 1e-8 * (1.0 + np.abs(sigma))
    if np.any(bad):
        def sigma_of_s(x):
            return c * x + np.sign(x) * np.abs(x) ** (2 * n - 1)
        for idx in np.argwhere(bad):
            idx = tuple(idx) if s.ndim > 1 else idx[0]
            sig = sigma[idx]
            if sig == 0.0:
                s[idx] = 0.0
                continue
            lo, hi = 0.0, np.sign(sig)
            step_b = 1.0
            while sigma_of_s(hi) * sig < 0 or abs(sigma_of_s(hi)) < abs(sig):
                step_b *= 2.0
                hi = np.sign(sig) * step_b
            a, b = (lo, hi) if sig > 0 else (hi, lo)
            s[idx] = brentq(lambda x: sigma_of_s(x) - sig, a, b, xtol=1e-14, rtol=1e-14)

    return s


def G_of_C(C, F, c, n):
    """G(C) = sum_i s(F_i + C). Root of G gives the periodic ground state."""
    return np.sum(slope_from_stress(F + C, c, n))


def solve_ground_state(f, c, n, C_bracket=None):
    """
    Full construction, steps 2-7 of the algorithm (step 1, drawing f, is
    assumed already done by the caller).

    Parameters
    ----------
    f : array, quenched random force (will be re-centered to zero mean)
    c : harmonic elastic constant (c >= 0)
    n : anharmonicity exponent (n > 1)

    Returns
    -------
    u : reconstructed interface (zero mean)
    s : slopes
    C : the scalar root
    F : cumulative force array
    """
    f = np.asarray(f, dtype=float)
    L = f.size

    # Step 2: enforce zero mean disorder
    f = f - f.mean()

    # Step 3: cumulative force F_i = sum_{j<i} f_j
    F = np.concatenate(([0.0], np.cumsum(f)))[:-1]  # F_0=0, F_i = f_0+...+f_{i-1}

    # Step 4: solve G(C) = 0 for C, using monotonicity to bracket the root
    if C_bracket is None:
        span = np.max(np.abs(F)) + 10 * np.std(f) + 10.0
        C_bracket = (-span - 10, span + 10)
    lo, hi = C_bracket
    # ensure sign change, expanding if necessary (G is strictly increasing in C)
    glo, ghi = G_of_C(lo, F, c, n), G_of_C(hi, F, c, n)
    while glo > 0:
        lo *= 2.0
        glo = G_of_C(lo, F, c, n)
    while ghi < 0:
        hi *= 2.0
        ghi = G_of_C(hi, F, c, n)
    C = brentq(lambda C: G_of_C(C, F, c, n), lo, hi, xtol=1e-12, rtol=1e-12)

    # Step 5: slopes
    s = slope_from_stress(F + C, c, n)

    # Step 6: reconstruct interface u_i = sum_{j<i} s_j
    u = np.concatenate(([0.0], np.cumsum(s)))[:-1]

    # Step 7: remove mean height
    u = u - u.mean()

    return u, s, C, F


def solve_ground_state_free(f, c, n):
    """
    Ground state for FREE (open) boundary conditions, instead of periodic.

    For an open chain, varying H at the two end sites gives stress-free
    boundary conditions sigma_{-1} = sigma_{L-1} = 0 (no bond beyond the
    ends). Solving the recursion sigma_k - sigma_{k-1} = f_k from the left
    boundary gives, with NO free integration constant:

        sigma_k = -sum_{j=0}^{k} f_j,   k = 0, ..., L-2  (L-1 bonds for L sites)

    The right-boundary condition then requires sum_j f_j = 0 for a
    consistent (mechanically equilibrated) solution to exist at all -- a
    free-free chain can only be static if the net applied force is zero.
    So we still enforce zero-mean disorder here, but for a different (and
    now unavoidable) reason than in the periodic case. Given that, C is
    identically zero: no root-finding step is needed at all.

    Returns
    -------
    u      : reconstructed interface, u[0] = 0, length L (zero mean removed)
    s      : slopes, length L-1
    F_full : cumulative sums sum_{j=0}^{k} f_j, k=0,...,L-1 (for reference)
    """
    f = np.asarray(f, dtype=float)

    # Required for a free-free static equilibrium to exist (see docstring).
    f = f - f.mean()

    F_full = np.cumsum(f)          # F_full[k] = sum_{j=0}^{k} f_j
    sigma = -F_full[:-1]           # sigma_k for k = 0,...,L-2 (L-1 bonds)

    s = slope_from_stress(sigma, c, n)

    u = np.concatenate(([0.0], np.cumsum(s)))   # u[0]=0, length L
    u = u - u.mean()

    return u, s, F_full


# ----------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    L = 2000
    Delta = 1.0
    c = 1.0
    n = 2.0  # quartic anharmonic term |s|^4/4

    # Step 1: generate disorder
    f = rng.normal(0.0, np.sqrt(Delta), size=L)

    u, s, C, F = solve_ground_state(f, c, n)

    # sanity checks
    periodic_closure = u[-1] + s[-1] - u[0]  # should be ~0 i.e. u_L == u_0
    residual = np.sum(s)
    print(f"L = {L}, c = {c}, n = {n}")
    print(f"solved C = {C:.6f}")
    print(f"sum of slopes (periodicity check, should be ~0): {residual:.3e}")
    print(f"mean height (should be ~0): {u.mean():.3e}")

    # check force balance residual: sigma_{i+1}-sigma_i = f_i  (using the
    # zero-mean force, since that's what solve_ground_state actually enforces)
    f_centered = f - f.mean()
    sigma = c * s + np.sign(s) * np.abs(s) ** (2 * n - 1)
    lhs = np.roll(sigma, -1) - sigma
    fb_residual = np.max(np.abs(lhs - f_centered))
    print(f"max force-balance residual: {fb_residual:.3e}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(u, lw=0.8, color="#1f77b4")
    axes[0].set_ylabel(r"$u_i$")
    axes[0].set_title(f"ALM ground state  (L={L}, c={c}, n={n}, $\\Delta$={Delta})")

    axes[1].plot(s, lw=0.6, color="#d62728")
    axes[1].set_ylabel(r"$s_i$")

    axes[2].plot(F, lw=0.6, color="#2ca02c", label=r"$F_i$")
    axes[2].axhline(-C, color="k", ls="--", lw=1, label=r"$-C$")
    axes[2].set_ylabel(r"$F_i$")
    axes[2].set_xlabel("site $i$")
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("alm_ground_state.png", dpi=150)
    print("Saved plot to alm_ground_state.png")
