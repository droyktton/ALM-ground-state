"""
Command-line tool: disorder-averaged structure factor for the ALM
====================================================================

Computes, from the same set of disorder samples:
  - S(q)     : disorder-averaged structure factor, and the fitted
               spectral roughness exponent zeta_s from S(q) ~ q^{-(1+2*zeta_s)}
  - W^2(L)   : interface width squared, W^2 = < (1/L) sum_i (u_i - mean(u))^2 >,
               computed two independent ways as a consistency check:
                 1. directly in real space from each sample's u_i
                 2. via Parseval's theorem, summing over the same S(q) used above

Usage
-----
    python3 run_structure_factor.py -L 4096 -n 2.0
    python3 run_structure_factor.py -L 8192 -n 3.0 -c 1.0 --delta 1.0 --samples 300
    python3 run_structure_factor.py -L 4096 -n 2.0 --qfrac 0.1 -o my_plot.png --no-show

Requires alm.py in the same folder (it provides solve_ground_state).
"""

import argparse
import sys
import time

import numpy as np
from scipy.stats import linregress
from scipy.fft import dct

from alm import solve_ground_state, solve_ground_state_free, sample_disorder


def structure_factor_average(L, c, n, Delta, n_samples, rng, dist="gaussian", bc="periodic"):
    """
    Returns
    -------
    q      : positive wavevectors
                 periodic: q_k = 2*pi*k/L, k=1,...,L//2   (shape L//2)
                 free:     q_k = pi*k/L,   k=1,...,L-1     (shape L-1)
    S      : disorder-averaged structure factor S(q) = < |coef(q)|^2 > / L
    W2_avg : disorder-averaged interface width squared,
             W2 = < (1/L) * sum_i (u_i - mean(u))^2 >
             computed directly in real space from the *same* samples used for S(q).

    bc="periodic" uses the exact periodic-BC solver and an FFT (rfft), as
    before. bc="free" uses the free (open) BC solver -- where C is exactly
    zero, no root-finding needed -- and a type-II orthonormal DCT instead of
    an FFT, since u(x) is not periodic for a free chain: feeding it into an
    FFT would introduce a spurious boundary discontinuity that contaminates
    the low-q power-law fit (Fourier power of a jump decays only as q^-2,
    which can dominate the genuine small-q behavior for zeta > 0.5).
    """
    if bc == "periodic":
        S_accum = np.zeros(L // 2)
        W2_accum = 0.0
        for _ in range(n_samples):
            f = sample_disorder(L, Delta, rng, dist=dist)
            u, s, C, F = solve_ground_state(f, c, n)   # u already has zero mean (step 7)

            u_hat = np.fft.rfft(u)
            power = (np.abs(u_hat) ** 2) / L
            S_accum += power[1:L // 2 + 1]   # drop k=0 mode (mean, exactly zero)

            W2_accum += np.mean(u ** 2)      # real-space width^2 for this sample

        S_avg = S_accum / n_samples
        W2_avg = W2_accum / n_samples
        k = np.arange(1, L // 2 + 1)
        q = 2 * np.pi * k / L
        return q, S_avg, W2_avg

    elif bc == "free":
        S_accum = np.zeros(L - 1)
        W2_accum = 0.0
        for _ in range(n_samples):
            f = sample_disorder(L, Delta, rng, dist=dist)
            u, s, F_full = solve_ground_state_free(f, c, n)   # u already zero-mean

            coef = dct(u, type=2, norm="ortho")
            power = (coef ** 2) / L
            S_accum += power[1:]   # drop k=0 mode (mean, exactly zero)

            W2_accum += np.mean(u ** 2)

        S_avg = S_accum / n_samples
        W2_avg = W2_accum / n_samples
        k = np.arange(1, L)
        q = np.pi * k / L
        return q, S_avg, W2_avg

    else:
        raise ValueError(f"Unknown bc '{bc}'. Choose 'periodic' or 'free'.")


def w2_from_S(q, S, L, bc="periodic"):
    """
    Parseval-theorem cross-check: reconstruct W^2 from the structure factor
    alone.

    periodic (rfft, mirrored negative frequencies of a real signal):
        W^2 = (1/L) * [ 2*sum_{k=1}^{L/2-1} S(q_k) + S(q_{L/2}) ]   (Nyquist counted once)

    free (orthonormal DCT-II, already one-sided/real, no mirrored modes):
        W^2 = sum_{k=1}^{L-1} S(q_k)                                 (no doubling, no 1/L factor
                                                                       -- already absorbed in S's
                                                                       definition for this case)
    """
    if bc == "periodic":
        is_even = (L % 2 == 0)
        if is_even:
            S_no_nyquist = S[:-1]
            S_nyquist = S[-1]
            total = 2.0 * np.sum(S_no_nyquist) + S_nyquist
        else:
            total = 2.0 * np.sum(S)
        return total / L
    elif bc == "free":
        return np.sum(S)
    else:
        raise ValueError(f"Unknown bc '{bc}'. Choose 'periodic' or 'free'.")


def fit_zeta_s(q, S, q_frac_max=0.15, q_min=None, q_max=None, kmin=None, kmax=None):
    """
    Fit S(q) ~ q^{-(1+2*zeta_s)} via log-log linear regression.

    The fit window can be specified three ways (checked in this priority
    order):
      1. kmin/kmax   : explicit mode-index bounds (1-based, inclusive of kmin,
                        exclusive of kmax+1) -- use this to hold a *fixed
                        absolute q-window* across different L, for finite-size
                        scans (q = 2*pi*k/L, so fixing k bounds while L varies
                        actually changes the physical q-window -- to fix the
                        physical window across L, use q_min/q_max instead).
      2. q_min/q_max : explicit absolute q bounds -- the natural choice for
                        finite-size scans: pick one physical q-window and use
                        the *same* q_min, q_max regardless of L.
      3. q_frac_max  : (default/back-compat) fraction of the L//2 available
                        modes, starting from k=1. Note the resulting q_max is
                        ~ pi * q_frac_max, independent of L, but q_min is
                        always 2*pi/L -- i.e. this does NOT hold a fixed
                        window across L on the IR side.

    Returns zeta_s, its standard error, the full regression result, and the
    (idx_min, idx_max) slice actually used (for plotting / bookkeeping).
    """
    if kmin is not None or kmax is not None:
        idx_min = 0 if kmin is None else (kmin - 1)
        idx_max = len(q) if kmax is None else kmax
    elif q_min is not None or q_max is not None:
        idx_min = 0 if q_min is None else int(np.searchsorted(q, q_min, side="left"))
        idx_max = len(q) if q_max is None else int(np.searchsorted(q, q_max, side="right"))
    else:
        idx_min = 0
        idx_max = max(4, int(len(q) * q_frac_max))

    idx_min = max(0, idx_min)
    idx_max = min(len(q), idx_max)
    if idx_max - idx_min < 4:
        raise ValueError(
            f"Fit window too narrow ({idx_max - idx_min} points, q range "
            f"[{q[idx_min] if idx_min < len(q) else float('nan')}, "
            f"{q[idx_max-1] if idx_max > 0 else float('nan')}]); "
            f"need at least 4 points. Widen q_min/q_max, kmin/kmax, or qfrac."
        )

    log_q = np.log(q[idx_min:idx_max])
    log_S = np.log(S[idx_min:idx_max])
    res = linregress(log_q, log_S)
    zeta_s = -(res.slope + 1) / 2
    zeta_s_err = res.stderr / 2
    return zeta_s, zeta_s_err, res, (idx_min, idx_max)


def main():
    parser = argparse.ArgumentParser(
        description="Disorder-averaged structure factor S(q) and spectral "
                    "roughness exponent zeta_s for the Anharmonic Larkin Model."
    )
    parser.add_argument("-L", type=int, required=True,
                        help="System size (number of sites). Should be even.")
    parser.add_argument("-n", type=float, required=True,
                        help="Anharmonicity exponent n (>1). n=2 -> quartic term |s|^4/4.")
    parser.add_argument("-c", type=float, default=1.0,
                        help="Harmonic elastic constant c >= 0 (default: 1.0).")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Disorder variance Delta (default: 1.0).")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of disorder realizations to average over (default: 200).")
    parser.add_argument("--qfrac", type=float, default=0.15,
                        help="Fraction of low-q modes used for the power-law fit "
                             "(default: 0.15). Ignored if --qmin/--qmax or "
                             "--kmin/--kmax are given.")
    parser.add_argument("--qmin", type=float, default=None,
                        help="Absolute lower bound of the fit window in q. "
                             "Use together with --qmax to fix the SAME physical "
                             "q-window across different L (recommended for "
                             "finite-size scans).")
    parser.add_argument("--qmax", type=float, default=None,
                        help="Absolute upper bound of the fit window in q.")
    parser.add_argument("--kmin", type=int, default=None,
                        help="Mode-index lower bound (1-based) of the fit window. "
                             "Overrides --qmin/--qmax/--qfrac if given.")
    parser.add_argument("--kmax", type=int, default=None,
                        help="Mode-index upper bound (inclusive) of the fit window.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("--dist", type=str, choices=["gaussian", "uniform", "bimodal"],
                        default="gaussian",
                        help="Disorder distribution shape, all matched to the same "
                             "variance Delta (default: gaussian).")
    parser.add_argument("--bc", type=str, choices=["periodic", "free"], default="periodic",
                        help="Boundary conditions: 'periodic' (default, root-find C, "
                             "FFT) or 'free' (open chain, C=0 exactly, no root-find, "
                             "uses a DCT since u(x) is not periodic).")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output PNG path (default: sf_L{L}_n{n}.png in current directory).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting entirely (just print the fit result).")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot to file without opening a display window.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to also dump the raw (q, S(q)) data as CSV.")
    args = parser.parse_args()

    if args.L % 2 != 0:
        print("Warning: L is odd; even L is recommended for a clean rfft/DCT spectrum.",
              file=sys.stderr)

    out_path = args.output or f"sf_L{args.L}_n{args.n}.png"

    rng = np.random.default_rng(args.seed)

    print(f"Averaging structure factor over {args.samples} samples "
          f"(L={args.L}, c={args.c}, n={args.n}, Delta={args.delta}, "
          f"dist={args.dist}, bc={args.bc}) ...")
    t0 = time.time()
    q, S, W2_direct = structure_factor_average(args.L, args.c, args.n, args.delta, args.samples,
                                                rng, dist=args.dist, bc=args.bc)
    print(f"done in {time.time() - t0:.2f} s")

    zeta_s, zeta_s_err, res, (idx_min, idx_max) = fit_zeta_s(
        q, S, q_frac_max=args.qfrac, q_min=args.qmin, q_max=args.qmax,
        kmin=args.kmin, kmax=args.kmax,
    )
    print(f"fit window: k in [{idx_min + 1}, {idx_max}]  "
          f"(q in [{q[idx_min]:.6g}, {q[idx_max - 1]:.6g}]), "
          f"{idx_max - idx_min} modes")
    print(f"fitted small-q slope   = {res.slope:.4f} +/- {res.stderr:.4f}")
    print(f"spectral roughness zeta_s = {zeta_s:.4f} +/- {zeta_s_err:.4f}")
    print(f"R^2 of log-log fit     = {res.rvalue**2:.5f}")

    W2_parseval = w2_from_S(q, S, args.L, bc=args.bc)
    rel_diff = abs(W2_direct - W2_parseval) / W2_direct
    print(f"W^2(L={args.L}) [real space, direct]  = {W2_direct:.6f}")
    print(f"W^2(L={args.L}) [from S(q), Parseval]  = {W2_parseval:.6f}")
    print(f"relative difference (consistency check) = {rel_diff:.2e}")

    # Single machine-readable line for easy parsing in shell scripts / pipelines.
    # Format: key=value pairs, space-separated, no spaces within a value.
    print(f"RESULT L={args.L} n={args.n} c={args.c} delta={args.delta} "
          f"samples={args.samples} dist={args.dist} bc={args.bc} "
          f"zeta_s={zeta_s:.6f} zeta_s_err={zeta_s_err:.6f} "
          f"W2_direct={W2_direct:.6f} W2_parseval={W2_parseval:.6f} "
          f"qmin_fit={q[idx_min]:.6g} qmax_fit={q[idx_max - 1]:.6g} "
          f"kmin_fit={idx_min + 1} kmax_fit={idx_max}")

    if args.csv:
        np.savetxt(args.csv, np.column_stack([q, S]), delimiter=",",
                   header="q,S(q)", comments="")
        print(f"Saved raw data to {args.csv}")

    if not args.no_plot:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.loglog(q, S, '.', ms=3, alpha=0.5, color="#1f77b4", label="data (disorder avg.)")

        q_fit = q[idx_min:idx_max]
        S_fit_line = np.exp(res.intercept) * q_fit ** res.slope
        ax.loglog(q_fit, S_fit_line, 'r-', lw=2,
                  label=fr"fit: $\zeta_s={zeta_s:.3f}\pm{zeta_s_err:.3f}$")

        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$S(q)$")
        ax.set_title(f"ALM structure factor  (L={args.L}, c={args.c}, n={args.n}, "
                    f"$\\Delta$={args.delta}, bc={args.bc}, {args.samples} samples)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        plt.tight_layout()

        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
