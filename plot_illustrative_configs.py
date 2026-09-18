"""
Illustrative ALM ground-state configurations for two values of p = 1/(2n-1),
under one or both boundary conditions (periodic / free).

Plots, for each p (and each requested bc):
  - [h(x) - mean(h)] / sigma_h   vs   x/L      (normalized interface shape)
  - m/sigma_m, m = dh/dx = s(x)  vs   x/L      (normalized slope field)

where sigma_h = sqrt(<[h-hbar]^2>) and sigma_m = sqrt(<m^2>) (m already has
zero mean: exactly by construction for periodic bc, and because the free-bc
disorder is also re-centered before solving), both computed for that single
realization. The same disorder realization f is reused across boundary
conditions (for a given p) so the periodic vs. free comparison isn't
confounded by different noise draws.

Uses the c=0 (pure anharmonic) case, since p = 1/(2n-1) is the exponent in
the c=0 constitutive relation s = sign(sigma) |sigma|^p.

Usage
-----
    python3 plot_illustrative_configs.py --p 0.2 4.0 -L 8192
    python3 plot_illustrative_configs.py --p 0.2 4.0 -L 16384 --seed 3 --no-show
    python3 plot_illustrative_configs.py --p 0.2 4.0 --bc periodic free
    python3 plot_illustrative_configs.py --p 0.2 4.0 --bc periodic free --combined
"""

import argparse

import numpy as np

from alm import solve_ground_state, solve_ground_state_free


def n_from_p(p):
    """Invert p = 1/(2n-1)  ->  n = (1 + 1/p) / 2."""
    return (1.0 + 1.0 / p) / 2.0


BC_STYLE = {"periodic": dict(ls="-"), "free": dict(ls="--")}


def solve_config(f, c, n, bc):
    """Dispatch to the periodic or free solver and return (u, s)."""
    if bc == "periodic":
        u, s, C, F = solve_ground_state(f, c, n)
    elif bc == "free":
        u, s, F_full = solve_ground_state_free(f, c, n)
    else:
        raise ValueError(f"Unknown bc '{bc}'. Choose 'periodic' or 'free'.")
    return u, s


def main():
    parser = argparse.ArgumentParser(
        description="Plot illustrative ALM configurations (normalized height "
                    "and slope vs x/L) for given values of p=1/(2n-1)."
    )
    parser.add_argument("--p", type=float, nargs="+", default=[0.2, 4.0],
                        help="Values of p=1/(2n-1) to illustrate (default: 0.2 4.0).")
    parser.add_argument("-L", type=int, default=8192,
                        help="System size (default: 8192).")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Disorder variance Delta (default: 1.0).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("-o", "--output", type=str, default="illustrative_configs.png",
                        help="Output plot path.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a display window.")
    parser.add_argument("--combined", action="store_true",
                        help="Overlay all p values on the same two panels "
                             "(vertical, 2 rows x 1 column) instead of one "
                             "column of panels per p.")
    parser.add_argument("--bc", type=str, nargs="+", choices=["periodic", "free"],
                        default=["periodic"],
                        help="Boundary condition(s) to show (default: "
                             "periodic). Pass both to compare them: "
                             "--bc periodic free. Each bc is drawn with its "
                             "own linestyle (solid=periodic, dashed=free), "
                             "reusing the same disorder realization per p.")
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.combined:
        fig, (ax_h, ax_s) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        colors = plt.get_cmap("tab10").colors

        for i, p in enumerate(args.p):
            n = n_from_p(p)
            rng = np.random.default_rng(args.seed)
            f = rng.normal(0.0, np.sqrt(args.delta), size=args.L)
            color = colors[i % len(colors)]

            for bc in args.bc:
                u, s = solve_config(f, c=0.0, n=n, bc=bc)

                sigma_h = np.sqrt(np.mean(u ** 2))
                sigma_m = np.sqrt(np.mean(s ** 2))  # zero-mean by construction
                x_over_L_h = np.arange(u.size) / args.L
                x_over_L_s = np.arange(s.size) / args.L
                ls = BC_STYLE[bc]["ls"]
                bc_tag = f", {bc}" if len(args.bc) > 1 else ""
                label = fr"$p={p:g}$ ($n={n:.4g}${bc_tag})"

                ax_h.plot(x_over_L_h, u / sigma_h, lw=0.7, ls=ls, color=color, label=label)
                ax_s.plot(x_over_L_s, s / sigma_m, lw=0.5, ls=ls, color=color, label=label)

        ax_h.set_ylabel(r"$[h(x)-\bar h]/\sigma_h$")
        ax_h.grid(True, ls=":", alpha=0.4)
        ax_h.legend(fontsize=9)

        ax_s.set_xlabel(r"$x/L$")
        ax_s.set_ylabel(r"$m/\sigma_m$")
        ax_s.grid(True, ls=":", alpha=0.4)
        ax_s.legend(fontsize=9)

        fig.suptitle(f"Illustrative ALM configurations  (c=0, L={args.L}, "
                    f"$\\Delta$={args.delta}, seed={args.seed})")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")

        if not args.no_show:
            plt.show()
        return

    n_p = len(args.p)
    fig, axes = plt.subplots(2, n_p, figsize=(6 * n_p, 7), sharex=True)
    if n_p == 1:
        axes = axes.reshape(2, 1)

    bc_colors = {"periodic": "#1f77b4", "free": "#d62728"}
    bc_colors_s = {"periodic": "#2ca02c", "free": "#ff7f0e"}

    for col, p in enumerate(args.p):
        n = n_from_p(p)
        rng = np.random.default_rng(args.seed)
        f = rng.normal(0.0, np.sqrt(args.delta), size=args.L)

        ax_h = axes[0, col]
        ax_s = axes[1, col]

        for bc in args.bc:
            u, s = solve_config(f, c=0.0, n=n, bc=bc)

            sigma_h = np.sqrt(np.mean(u ** 2))
            sigma_m = np.sqrt(np.mean(s ** 2))  # zero-mean by construction
            x_over_L_h = np.arange(u.size) / args.L
            x_over_L_s = np.arange(s.size) / args.L
            ls = BC_STYLE[bc]["ls"]
            label = bc if len(args.bc) > 1 else None

            ax_h.plot(x_over_L_h, u / sigma_h, lw=0.7, ls=ls,
                     color=bc_colors[bc], label=label)
            ax_s.plot(x_over_L_s, s / sigma_m, lw=0.5, ls=ls,
                     color=bc_colors_s[bc], label=label)

        ax_h.set_title(fr"$p={p:g}$  ($n={n:.4g}$)")
        ax_h.set_ylabel(r"$[h(x)-\bar h]/\sigma_h$")
        ax_h.grid(True, ls=":", alpha=0.4)

        ax_s.set_xlabel(r"$x/L$")
        ax_s.set_ylabel(r"$m/\sigma_m$")
        ax_s.grid(True, ls=":", alpha=0.4)

        if len(args.bc) > 1:
            ax_h.legend(fontsize=8)
            ax_s.legend(fontsize=8)

    fig.suptitle(f"Illustrative ALM configurations  (c=0, L={args.L}, "
                f"$\\Delta$={args.delta}, seed={args.seed})")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
