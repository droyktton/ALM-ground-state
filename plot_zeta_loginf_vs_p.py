"""
For each n (equivalently p = 1/(2n-1)) and boundary condition, fit

    zeta_s(L) = zeta_inf - a / ln(L)

by weighted least squares (weights = 1/zeta_s_err^2) across the L sweep,
and plot the extrapolated zeta_inf(p) against the piecewise theory curve.

Why log(L) and not a power of L
--------------------------------
This follows up on plot_zeta_offset_vs_p.py / plot_zeta_gap_vs_L.py, which
found that at p=0.5 (the kink of the piecewise theory) the gap between
zeta_theory and zeta_s(L) shrinks only very slowly with L and looked like
it might not close at all. Interactively fitting zeta_s(L) vs 1/L and vs
1/sqrt(L) at p=0.5 both extrapolated to ~1.47, many sigma away from the
theory value 1.5, with a poor fit to the largest-L points in either case
(R^2 ~ 0.4-0.5). Switching to 1/ln(L) fixed both problems: it fits the
full L-range much better (R^2 ~ 0.97) and extrapolates to 1.498 +/- 0.005,
i.e. consistent with 1.5 to <1 sigma. That's the standard signature of a
*marginal* point with logarithmic finite-size corrections, which p=0.5
plausibly is (it's the boundary between the two branches of the piecewise
theory) -- so this script applies the same 1/ln(L) extrapolation at every
p in the sweep, not just p=0.5, to see where else it matters.

zeta_theory(p) is the piecewise theory curve used in plot_zeta_vs_n.py /
plot_zeta_offset_vs_p.py:
    zeta = 1 + p/2 + 1/4   (p < 0.5)
    zeta = 1.5             (p >= 0.5)

Expected CSV columns (as written by run_structure_factor.py's RESULT line):
    bc,L,n,c,delta,samples,zeta_s,zeta_s_err,W2_direct,W2_parseval

Usage
-----
    python3 plot_zeta_loginf_vs_p.py --csv scan_results.csv
    python3 plot_zeta_loginf_vs_p.py --csv scan_results.csv --no-show -o zeta_loginf_vs_p.png
    python3 plot_zeta_loginf_vs_p.py --csv scan_results.csv --min-L 32768  # drop small-L points from the fit
"""

import argparse

import numpy as np
import pandas as pd


def zeta_theory_fn(p):
    """Piecewise theory curve: zeta = 1 + p/2 + 1/4 (p<0.5), zeta = 1.5 (p>=0.5)."""
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.5, 1.0 + p / 2.0 + 0.25, 1.5)


def weighted_loginf_fit(L, zeta_s, zeta_s_err):
    """Weighted least-squares fit of zeta_s vs x=1/ln(L): zeta_s = intercept + slope*x.

    Returns intercept (zeta_inf as L->infinity), its stderr, slope, slope
    stderr, and the fit's reduced chi^2 (goodness of fit of the 1/ln(L)
    form itself, dof = n_points - 2).
    """
    x = 1.0 / np.log(L)
    w = 1.0 / zeta_s_err**2
    Sw = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * zeta_s)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * zeta_s)
    denom = Sw * Sxx - Sx**2

    slope = (Sw * Sxy - Sx * Sy) / denom
    intercept = (Sxx * Sy - Sx * Sxy) / denom
    intercept_err = np.sqrt(Sxx / denom)
    slope_err = np.sqrt(Sw / denom)

    n = len(L)
    dof = n - 2
    if dof > 0:
        resid = zeta_s - (intercept + slope * x)
        chi2 = np.sum(w * resid**2)
        chi2_red = chi2 / dof
    else:
        chi2_red = np.nan

    return intercept, intercept_err, slope, slope_err, chi2_red, n


def summarize_bc(df_bc, bc_label, min_L=None):
    rows = []
    for n_val, d in df_bc.groupby("n"):
        d = d.sort_values("L")
        if min_L is not None:
            d = d[d["L"] >= min_L]
        if len(d) < 3:
            print(f"[{bc_label}] n={n_val:>5}: skipped, only {len(d)} L-points "
                  f"(need >= 3 for a 2-parameter fit)")
            continue

        zeta_inf, zeta_inf_err, slope, slope_err, chi2_red, n_pts = weighted_loginf_fit(
            d["L"].values.astype(float), d["zeta_s"].values, d["zeta_s_err"].values)
        p_val = 1.0 / (2.0 * n_val - 1.0)
        theory = float(zeta_theory_fn(p_val))
        offset = theory - zeta_inf
        sigma = offset / zeta_inf_err
        rows.append(dict(n=n_val, p=p_val, zeta_inf=zeta_inf, zeta_inf_err=zeta_inf_err,
                         slope=slope, chi2_red=chi2_red, n_L=n_pts, zeta_theory=theory,
                         offset=offset, sigma=sigma))
        print(f"[{bc_label}] n={n_val:>5} (p={p_val:.3f}): "
              f"zeta_inf = {zeta_inf:.5f} +/- {zeta_inf_err:.5f}  "
              f"chi2_red = {chi2_red:.2f} ({n_pts} L-points)  "
              f"offset = {offset:+.4f} ({sigma:+.1f} sigma)")
    return pd.DataFrame(rows).sort_values("p")


def plot_bc_panel(ax, summary, bc_label):
    ax.errorbar(summary["p"], summary["offset"], yerr=summary["zeta_inf_err"],
                fmt='ko-', ms=6, lw=1.5, capsize=3)
    for _, row in summary.iterrows():
        ax.annotate(fr"${row['sigma']:+.1f}\sigma$",
                    (row["p"], row["offset"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="gray")
    ax.axhline(0.0, color="b", ls="--", lw=1.2, alpha=0.7,
               label="theory (no offset)")
    ax.set_xlabel(r"$p = 1/(2n-1)$")
    ax.set_ylabel(r"$\zeta_{\rm theory}(p) - \zeta_\infty(p)$")
    ax.set_title(f"bc = {bc_label}")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)


def main():
    parser = argparse.ArgumentParser(
        description="Weighted 1/ln(L) extrapolation of zeta_s(L) to L->infinity "
                    "per n, and offset of the extrapolated value vs theory, "
                    "plotted against p."
    )
    parser.add_argument("--csv", type=str, default="scan_results.csv",
                        help="Input CSV path (default: scan_results.csv).")
    parser.add_argument("-o", "--output", type=str, default="zeta_loginf_vs_p.png",
                        help="Output plot path.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a display window.")
    parser.add_argument("--bc", type=str, choices=["periodic", "free", "both"],
                        default="both",
                        help="Which boundary condition(s) to plot (default: "
                             "both, one panel each).")
    parser.add_argument("--min-L", type=float, default=None,
                        help="Exclude L below this value from the 1/ln(L) fit "
                             "(the fit is already an extrapolation to large L, "
                             "so this mainly matters if the smallest sizes are "
                             "badly off the log form).")
    parser.add_argument("--summary-csv", type=str, default=None,
                        help="Optional path to save the per-n summary table "
                             "(zeta_inf, chi2_red, offset, sigma) as its own CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("zeta_s", "zeta_s_err"):
        df[col] = pd.to_numeric(df[col])

    has_bc = "bc" in df.columns
    if has_bc and args.bc != "both":
        df = df[df["bc"] == args.bc]
        if df.empty:
            raise SystemExit(f"No rows with bc={args.bc!r} found in {args.csv}")

    global plt
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if has_bc:
        bc_values = [b for b in ["periodic", "free"] if b in df["bc"].unique()]
    else:
        bc_values = [None]

    n_panels = len(bc_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    axes = axes[0]

    summary_frames = []
    for ax, bc_val in zip(axes, bc_values):
        df_bc = df[df["bc"] == bc_val] if bc_val is not None else df
        bc_label = bc_val if bc_val is not None else "n/a"
        summary = summarize_bc(df_bc, bc_label, min_L=args.min_L)
        plot_bc_panel(ax, summary, bc_label)
        if bc_val is not None:
            summary = summary.assign(bc=bc_val)
        summary_frames.append(summary)

    fig.suptitle(r"$\zeta_{\rm theory}(p) - \zeta_\infty(p)$ "
                r"($\zeta_s(L) = \zeta_\infty - a/\ln L$ extrapolation) vs $p$")

    if args.summary_csv and summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(args.summary_csv, index=False)
        print(f"\nSaved summary table to {args.summary_csv}")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
