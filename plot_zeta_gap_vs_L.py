"""
Plot zeta_theory(p) - zeta_s(p, L)  vs  L, one line per p = 1/(2n-1), to see
how the structure-factor exponent zeta_s converges (or not) to the
theoretical prediction as the system size grows.

zeta_theory(p) is the piecewise theory curve used in plot_zeta_vs_n.py:
    zeta = 1 + p/2 + 1/4   (p < 0.5)
    zeta = 1.5             (p >= 0.5)

From a single scan_results.csv produced by a sweep over (bc, L, n) (e.g.
scan_sweep_bc.sh / slurm/scan_array.sbatch), with one panel per boundary
condition so free and periodic results can be compared side by side.

Expected CSV columns (as written by run_structure_factor.py's RESULT line):
    bc,L,n,c,delta,samples,zeta_s,zeta_s_err,W2_direct,W2_parseval

Usage
-----
    python3 plot_zeta_gap_vs_L.py --csv scan_results.csv
    python3 plot_zeta_gap_vs_L.py --csv scan_results.csv --no-show -o zeta_gap_vs_L.png
    python3 plot_zeta_gap_vs_L.py --csv scan_results.csv --bc periodic
"""

import argparse

import numpy as np
import pandas as pd


def zeta_theory_fn(p):
    """Piecewise theory curve: zeta = 1 + p/2 + 1/4 (p<0.5), zeta = 1.5 (p>=0.5)."""
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.5, 1.0 + p / 2.0 + 0.25, 1.5)


def plot_bc_panel(ax, df_bc, bc_label):
    ns = sorted(df_bc["n"].unique())
    cmap = plt.get_cmap("viridis")
    for i, n_val in enumerate(ns):
        d = df_bc[df_bc["n"] == n_val].sort_values("L")
        p_val = 1.0 / (2.0 * n_val - 1.0)
        gap = zeta_theory_fn(p_val) - d["zeta_s"]
        color = cmap(i / max(1, len(ns) - 1))
        ax.errorbar(d["L"], gap, yerr=d["zeta_s_err"], fmt='o-', ms=4, lw=1.2,
                    alpha=0.85, color=color, label=fr"$n$={n_val:g}, $p$={p_val:.3f}")

    ax.axhline(0.0, color="k", ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\zeta_{\rm theory}(p) - \zeta_s(p, L)$")
    ax.set_title(f"bc = {bc_label}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, ls=":", alpha=0.4)


def main():
    parser = argparse.ArgumentParser(
        description="Plot zeta_theory(p) - zeta_s(p,L) vs L, one line per p, "
                    "with one panel per boundary condition (free / periodic)."
    )
    parser.add_argument("--csv", type=str, default="scan_results.csv",
                        help="Input CSV path (default: scan_results.csv).")
    parser.add_argument("-o", "--output", type=str, default="zeta_gap_vs_L.png",
                        help="Output plot path.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a display window.")
    parser.add_argument("--bc", type=str, choices=["periodic", "free", "both"],
                        default="both",
                        help="Which boundary condition(s) to plot (default: "
                             "both, one panel each).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # W2 columns aren't needed here, but zeta_s/zeta_s_err can hit the same
    # long-fixed-point-notation parsing issue at large L/low n -- be safe.
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

    for ax, bc_val in zip(axes, bc_values):
        df_bc = df[df["bc"] == bc_val] if bc_val is not None else df
        bc_label = bc_val if bc_val is not None else "n/a"
        plot_bc_panel(ax, df_bc, bc_label)

    fig.suptitle(r"$\zeta_{\rm theory}(p) - \zeta_s(p, L)$ vs $L$ (piecewise theory)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
