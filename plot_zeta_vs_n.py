"""
Plot zeta_s(n, L) and zeta(n) from W^2(L) ~ L^{2*zeta} fits, against
p = 1/(2n-1), together with two theory curves:
    zeta = 1 + p/2                                  (whole range)
    zeta = 1 + p/2 + 1/4 (p<0.5), zeta = 1.5 (p>=0.5)  (piecewise)

All from a single scan_results.csv produced by a sweep over (bc, L, n)
(e.g. scan_sweep_bc.sh), with one panel per boundary condition so free and
periodic results can be compared side by side.

Expected CSV columns (as written by run_structure_factor.py's RESULT line,
collected e.g. by scan_sweep_bc.sh):
    bc,L,n,c,delta,samples,zeta_s,zeta_s_err,W2_direct,W2_parseval

If the CSV has no "bc" column (older single-bc sweeps), it is treated as a
single implicit boundary condition and plotted in one panel.

Usage
-----
    python3 plot_zeta_vs_n.py --csv scan_results.csv
    python3 plot_zeta_vs_n.py --csv scan_results.csv --Lmin 8192 -o zeta_vs_p.png
    python3 plot_zeta_vs_n.py --csv scan_results.csv --Lmin 8192 --no-show
    python3 plot_zeta_vs_n.py --csv scan_results.csv --xaxis n   # plot vs n instead
    python3 plot_zeta_vs_n.py --csv scan_results.csv --bc periodic  # single bc only
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import linregress


def fit_zeta_from_W2(df_n, L_min=None, L_max=None):
    """
    For a single n's data (multiple L rows), fit W^2(L) ~ L^{2*zeta} via
    log-log linear regression across the available L's (optionally
    restricted to [L_min, L_max] to exclude small-L finite-size points).

    Returns zeta, zeta_err, R^2, n_points_used (or None if not enough points).
    """
    d = df_n.sort_values("L")
    if L_min is not None:
        d = d[d["L"] >= L_min]
    if L_max is not None:
        d = d[d["L"] <= L_max]
    if len(d) < 2:
        return None

    log_L = np.log(d["L"].values.astype(float))
    log_W2 = np.log(d["W2_direct"].values.astype(float))
    res = linregress(log_L, log_W2)
    zeta = res.slope / 2.0
    zeta_err = res.stderr / 2.0
    return zeta, zeta_err, res.rvalue ** 2, len(d)


def zeta_theory_fn(p):
    """Piecewise theory curve: zeta = 1 + p/2 + 1/4 (p<0.5), zeta = 1.5 (p>=0.5)."""
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.5, 1.0 + p / 2.0 + 0.25, 1.5)


def plot_bc_panel(ax, df_bc, xcol, Lmin, Lmax, zeta_s_L, bc_label):
    """Fit zeta(n) from W^2(L) scaling and plot everything for one boundary
    condition (one subset of rows) onto a single axes."""

    # zeta from W^2(L) ~ L^{2 zeta}, fit per n across the L sweep
    rows = []
    for n_val, df_n in df_bc.groupby("n"):
        fit = fit_zeta_from_W2(df_n, L_min=Lmin, L_max=Lmax)
        if fit is None:
            print(f"[{bc_label}] n={n_val}: not enough L points for a W^2(L) fit, skipping.")
            continue
        zeta, zeta_err, R2, n_pts = fit
        p_val = 1.0 / (2.0 * n_val - 1.0)
        rows.append(dict(n=n_val, p=p_val, zeta_W2=zeta, zeta_W2_err=zeta_err,
                         R2=R2, n_points=n_pts))
        print(f"[{bc_label}] n={n_val:>5}: zeta_from_W2 = {zeta:.4f} +/- {zeta_err:.4f}  "
              f"(R^2={R2:.4f}, {n_pts} L-points)")

    zeta_W2_df = pd.DataFrame(rows).sort_values("n") if rows else pd.DataFrame(
        columns=["n", "p", "zeta_W2", "zeta_W2_err", "R2", "n_points"])

    # zeta_s(n, L) from the structure-factor fits -- plot all available L's
    # unless the caller asked to restrict to a single one via --zeta-s-L.
    if zeta_s_L is not None:
        Ls_to_plot = [zeta_s_L]
    else:
        Ls_to_plot = sorted(df_bc["L"].unique())

    cmap = plt.get_cmap("viridis")
    for i, L_val in enumerate(Ls_to_plot):
        d = df_bc[df_bc["L"] == L_val].sort_values(xcol)
        if d.empty:
            continue
        color = cmap(i / max(1, len(Ls_to_plot) - 1))
        ax.errorbar(d[xcol], d["zeta_s"], yerr=d["zeta_s_err"],
                    fmt='o-', ms=4, lw=1, alpha=0.7, color=color,
                    label=fr"$\zeta_s$, L={int(L_val)}")

    if not zeta_W2_df.empty:
        zeta_W2_df = zeta_W2_df.sort_values(xcol)
        ax.errorbar(zeta_W2_df[xcol], zeta_W2_df["zeta_W2"], yerr=zeta_W2_df["zeta_W2_err"],
                    fmt='ks', ms=7, lw=1.8, capsize=3, zorder=10,
                    label=r"$\zeta$ from $W^2(L)\sim L^{2\zeta}$")

    if xcol == "p":
        p_theory = np.linspace(df_bc["p"].min(), df_bc["p"].max(), 400)
        ax.plot(p_theory, 1.0 + p_theory / 2.0, 'b--', lw=1.8, zorder=10,
               label=r"theory: $\zeta = 1 + p/2$")
        ax.plot(p_theory, zeta_theory_fn(p_theory), 'r-', lw=2, zorder=11,
               label=r"theory: $\zeta=1{+}p/2{+}1/4$ ($p{<}0.5$), $\zeta=1.5$ ($p{\geq}0.5$)")
        ax.set_xlabel(r"$p = 1/(2n-1)$")
    else:
        n_theory = np.linspace(df_bc["n"].min(), df_bc["n"].max(), 400)
        p_theory = 1.0 / (2.0 * n_theory - 1.0)
        ax.plot(n_theory, 1.0 + p_theory / 2.0, 'b--', lw=1.8, zorder=10,
               label=r"theory: $\zeta = 1 + p/2$")
        ax.plot(n_theory, zeta_theory_fn(p_theory), 'r-', lw=2, zorder=11,
               label=r"theory: $\zeta=1{+}p/2{+}1/4$ ($p{<}0.5$), $\zeta=1.5$ ($p{\geq}0.5$)")
        ax.set_xlabel("n")

    ax.set_ylabel(r"$\zeta$")
    ax.set_title(f"bc = {bc_label}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, ls=":", alpha=0.4)

    return zeta_W2_df


def main():
    parser = argparse.ArgumentParser(
        description="Plot zeta_s(n,L) and zeta(n) from W^2(L) scaling, "
                    "from a scan_results.csv, with one panel per boundary "
                    "condition (free / periodic)."
    )
    parser.add_argument("--csv", type=str, default="scan_results.csv",
                        help="Input CSV path (default: scan_results.csv).")
    parser.add_argument("--Lmin", type=float, default=None,
                        help="Exclude L below this value from the W^2(L) fit "
                             "(recommended, to avoid small-L finite-size bias).")
    parser.add_argument("--Lmax", type=float, default=None,
                        help="Exclude L above this value from the W^2(L) fit.")
    parser.add_argument("-o", "--output", type=str, default="zeta_vs_n.png",
                        help="Output plot path.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a display window.")
    parser.add_argument("--zeta-s-L", type=float, default=None,
                        help="If given, only plot zeta_s(n) at this single L "
                             "(cleaner plot); default plots all L's present.")
    parser.add_argument("--summary-csv", type=str, default=None,
                        help="Optional path to save the fitted zeta(n) [from "
                             "W^2 scaling] table as its own CSV (includes a "
                             "'bc' column when multiple boundary conditions "
                             "are present).")
    parser.add_argument("--xaxis", type=str, choices=["p", "n"], default="p",
                        help="Plot vs p=1/(2n-1) (default -- makes the theory "
                             "curve the identity line zeta=p) or vs n directly.")
    parser.add_argument("--bc", type=str, choices=["periodic", "free", "both"],
                        default="both",
                        help="Which boundary condition(s) to plot (default: "
                             "both, one panel each).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    has_bc = "bc" in df.columns
    if has_bc and args.bc != "both":
        df = df[df["bc"] == args.bc]
        if df.empty:
            raise SystemExit(f"No rows with bc={args.bc!r} found in {args.csv}")

    group_cols = ["bc", "L", "n"] if has_bc else ["L", "n"]
    # Collapse any accidental duplicate rows by averaging.
    df = df.groupby(group_cols, as_index=False).mean(numeric_only=True)
    df["p"] = 1.0 / (2.0 * df["n"] - 1.0)

    global plt
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xcol = args.xaxis  # "p" or "n"

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
        zeta_W2_df = plot_bc_panel(ax, df_bc, xcol, args.Lmin, args.Lmax,
                                    args.zeta_s_L, bc_label)
        if bc_val is not None:
            zeta_W2_df = zeta_W2_df.assign(bc=bc_val)
        summary_frames.append(zeta_W2_df)

    fig.suptitle("Roughness exponent vs " + ("$p=1/(2n-1)$" if xcol == "p" else "anharmonicity n"))

    if args.summary_csv and summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(args.summary_csv, index=False)
        print(f"Saved W^2-scaling summary to {args.summary_csv}")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
