"""
For each n (equivalently p = 1/(2n-1)) and boundary condition, take the
zeta_s(L) values across the whole L sweep and test whether they're
consistent with a single constant (i.e. already converged, no residual
L-dependence) via a weighted mean and reduced chi^2. Then plot the offset
zeta_theory(p) - <zeta_s>_L  vs  p, with error bars, to see the shape of
the disagreement with the piecewise theory once finite-size scatter is
averaged out.

This follows up on plot_zeta_gap_vs_L.py: that plot shows the gap between
theory and zeta_s(p,L) barely moves across 7 decades of L (2048 to 2^21),
suggesting it's not a finite-size effect that will close as L grows, but a
real offset already present at the smallest L. This script quantifies
that with a proper weighted average + goodness-of-fit test instead of
eyeballing the flat lines.

zeta_theory(p) is the piecewise theory curve used in plot_zeta_vs_n.py:
    zeta = 1 + p/2 + 1/4   (p < 0.5)
    zeta = 1.5             (p >= 0.5)

Expected CSV columns (as written by run_structure_factor.py's RESULT line):
    bc,L,n,c,delta,samples,zeta_s,zeta_s_err,W2_direct,W2_parseval

Usage
-----
    python3 plot_zeta_offset_vs_p.py --csv scan_results.csv
    python3 plot_zeta_offset_vs_p.py --csv scan_results.csv --no-show -o zeta_offset_vs_p.png
"""

import argparse

import numpy as np
import pandas as pd


def zeta_theory_fn(p):
    """Piecewise theory curve: zeta = 1 + p/2 + 1/4 (p<0.5), zeta = 1.5 (p>=0.5)."""
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.5, 1.0 + p / 2.0 + 0.25, 1.5)


def weighted_mean_and_chi2(zeta_s, zeta_s_err):
    """Inverse-variance weighted mean of zeta_s across L, plus reduced
    chi^2 against the constant-value hypothesis (dof = N - 1)."""
    w = 1.0 / zeta_s_err**2
    mean = np.sum(w * zeta_s) / np.sum(w)
    mean_err = 1.0 / np.sqrt(np.sum(w))
    n = len(zeta_s)
    if n > 1:
        chi2 = np.sum(((zeta_s - mean) / zeta_s_err) ** 2)
        chi2_red = chi2 / (n - 1)
    else:
        chi2_red = np.nan
    return mean, mean_err, chi2_red, n


def summarize_bc(df_bc, bc_label):
    rows = []
    for n_val, d in df_bc.groupby("n"):
        mean, mean_err, chi2_red, n_pts = weighted_mean_and_chi2(
            d["zeta_s"].values, d["zeta_s_err"].values)
        p_val = 1.0 / (2.0 * n_val - 1.0)
        theory = float(zeta_theory_fn(p_val))
        offset = theory - mean
        rows.append(dict(n=n_val, p=p_val, zeta_mean=mean, zeta_mean_err=mean_err,
                         chi2_red=chi2_red, n_L=n_pts, zeta_theory=theory,
                         offset=offset))
        print(f"[{bc_label}] n={n_val:>5} (p={p_val:.3f}): "
              f"<zeta_s>_L = {mean:.5f} +/- {mean_err:.5f}  "
              f"chi2_red = {chi2_red:.2f} ({n_pts} L-points)  "
              f"offset = {offset:+.4f}")
    return pd.DataFrame(rows).sort_values("p")


def plot_bc_panel(ax, summary, bc_label):
    ax.errorbar(summary["p"], summary["offset"], yerr=summary["zeta_mean_err"],
                fmt='ko-', ms=6, lw=1.5, capsize=3)
    for _, row in summary.iterrows():
        ax.annotate(fr"$\chi^2_\nu$={row['chi2_red']:.1f}",
                    (row["p"], row["offset"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="gray")
    ax.axhline(0.0, color="b", ls="--", lw=1.2, alpha=0.7,
               label="theory (no offset)")
    ax.set_xlabel(r"$p = 1/(2n-1)$")
    ax.set_ylabel(r"$\zeta_{\rm theory}(p) - \langle\zeta_s\rangle_L$")
    ax.set_title(f"bc = {bc_label}")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)


def main():
    parser = argparse.ArgumentParser(
        description="Weighted-mean zeta_s across L per n, chi^2 test for "
                    "convergence, and offset vs theory plotted against p."
    )
    parser.add_argument("--csv", type=str, default="scan_results.csv",
                        help="Input CSV path (default: scan_results.csv).")
    parser.add_argument("-o", "--output", type=str, default="zeta_offset_vs_p.png",
                        help="Output plot path.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a display window.")
    parser.add_argument("--bc", type=str, choices=["periodic", "free", "both"],
                        default="both",
                        help="Which boundary condition(s) to plot (default: "
                             "both, one panel each).")
    parser.add_argument("--summary-csv", type=str, default=None,
                        help="Optional path to save the per-n summary table "
                             "(mean, chi2_red, offset) as its own CSV.")
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
        summary = summarize_bc(df_bc, bc_label)
        plot_bc_panel(ax, summary, bc_label)
        if bc_val is not None:
            summary = summary.assign(bc=bc_val)
        summary_frames.append(summary)

    fig.suptitle(r"$\zeta_{\rm theory}(p) - \langle\zeta_s\rangle_L$ (weighted "
                r"mean over $L$) vs $p$, with reduced $\chi^2$ per point")

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
