"""
Illustrative ALM ground-state configurations for two values of p = 1/(2n-1).

Plots, for each p:
  - [h(x) - mean(h)] / sigma_h   vs   x/L      (normalized interface shape)
  - m/sigma_m, m = dh/dx = s(x)  vs   x/L      (normalized slope field)

where sigma_h = sqrt(<[h-hbar]^2>) and sigma_m = sqrt(<m^2>) (m already has
zero mean, since sum_i s_i = 0 by the periodicity constraint), both computed
for that single realization.

Uses the c=0 (pure anharmonic) case, since p = 1/(2n-1) is the exponent in
the c=0 constitutive relation s = sign(sigma) |sigma|^p.

Usage
-----
    python3 plot_illustrative_configs.py --p 0.2 4.0 -L 8192
    python3 plot_illustrative_configs.py --p 0.2 4.0 -L 16384 --seed 3 --no-show
"""

import argparse

import numpy as np

from alm import solve_ground_state


def n_from_p(p):
    """Invert p = 1/(2n-1)  ->  n = (1 + 1/p) / 2."""
    return (1.0 + 1.0 / p) / 2.0


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

            u, s, C, F = solve_ground_state(f, c=0.0, n=n)

            sigma_h = np.sqrt(np.mean(u ** 2))
            sigma_m = np.sqrt(np.mean(s ** 2))  # s already zero-mean (sum s_i = 0)
            x_over_L = np.arange(args.L) / args.L
            color = colors[i % len(colors)]
            label = fr"$p={p:g}$ ($n={n:.4g}$)"

            ax_h.plot(x_over_L, u / sigma_h, lw=0.7, color=color, label=label)
            ax_s.plot(x_over_L, s / sigma_m, lw=0.5, color=color, label=label)

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

    for col, p in enumerate(args.p):
        n = n_from_p(p)
        rng = np.random.default_rng(args.seed)
        f = rng.normal(0.0, np.sqrt(args.delta), size=args.L)

        u, s, C, F = solve_ground_state(f, c=0.0, n=n)

        sigma_h = np.sqrt(np.mean(u ** 2))  # u already zero-mean by construction
        sigma_m = np.sqrt(np.mean(s ** 2))  # s already zero-mean (sum s_i = 0)
        x_over_L = np.arange(args.L) / args.L

        ax_h = axes[0, col]
        ax_h.plot(x_over_L, u / sigma_h, lw=0.7, color="#1f77b4")
        ax_h.set_title(fr"$p={p:g}$  ($n={n:.4g}$)")
        ax_h.set_ylabel(r"$[h(x)-\bar h]/\sigma_h$")
        ax_h.grid(True, ls=":", alpha=0.4)

        ax_s = axes[1, col]
        ax_s.plot(x_over_L, s / sigma_m, lw=0.5, color="#d62728")
        ax_s.set_xlabel(r"$x/L$")
        ax_s.set_ylabel(r"$m/\sigma_m$")
        ax_s.grid(True, ls=":", alpha=0.4)

    fig.suptitle(f"Illustrative ALM configurations  (c=0, L={args.L}, "
                f"$\\Delta$={args.delta}, seed={args.seed})")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
