#!/usr/bin/env python3
"""
3D plot of all delay-sum look directions (upper hemisphere grid).

Shows the discrete steering set used by ``hemisphere_power_map``; angular spacing
between grid points bounds peak-picking resolution. Physical main-lobe width is
also frequency-dependent (see printed summary).

Example:
  python plot_steering_grid.py --save steering_grid.png --no-show
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

import geometry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 3D beamformer steering grid")
    p.add_argument("--n-az", type=int, default=36, help="Azimuth bins (same as run_simulation default)")
    p.add_argument("--n-el", type=int, default=10, help="Elevation bins")
    p.add_argument(
        "--el-max",
        type=float,
        default=89.0,
        help="Max elevation (deg); grid is 0..el_max (upper hemisphere cap)",
    )
    p.add_argument("--mic-radius", type=float, default=0.03, help="Array radius (m) for mic markers")
    p.add_argument("--save", type=str, default="", help="Output image path (e.g. steering_grid.png)")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    az_g, el_g, dirs = geometry.upper_hemisphere_grid(args.n_az, args.n_el, el_max_deg=args.el_max)
    pts = dirs.reshape(-1, 3)
    el_flat = el_g.reshape(-1)
    az_step = 360.0 / args.n_az
    el_vals = np.linspace(0.0, args.el_max, args.n_el)
    el_step = float(np.mean(np.diff(el_vals))) if args.n_el > 1 else 0.0

    mic_xyz = geometry.mic_positions_circular_xy(8, args.mic_radius)

    # Rough narrowband horizontal-plane Rayleigh width for uniform ring (~ first null):
    # θ ≈ 0.61 λ / D  (radians). Use mid-band ~2 kHz as illustration.
    fs_typ = 48_000.0
    f_mid = 2000.0
    wavelength = geometry.C_SOUND / f_mid
    d_ap = 2.0 * args.mic_radius
    rayleigh_deg = float(np.degrees(0.61 * wavelength / d_ap))

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Unit hemisphere wireframe (z >= 0)
    uu = np.linspace(0.0, 2.0 * np.pi, 48)
    vv = np.linspace(0.0, np.pi / 2.0, 24)
    U, V = np.meshgrid(uu, vv)
    xs = np.cos(V) * np.cos(U)
    ys = np.cos(V) * np.sin(U)
    zs = np.sin(V)
    ax.plot_wireframe(xs, ys, zs, color="0.75", linewidth=0.4, alpha=0.5, rstride=2, cstride=3)

    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=el_flat,
        cmap="viridis",
        s=36,
        depthshade=True,
        edgecolors="k",
        linewidths=0.2,
        label="Steering directions",
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label("Elevation (deg)")

    ax.scatter(mic_xyz[:, 0], mic_xyz[:, 1], mic_xyz[:, 2], c="crimson", s=80, marker="o", label="Mics (z=0)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    lim = 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, lim)
    ax.set_box_aspect((1, 1, 0.55))

    n_dirs = pts.shape[0]
    title = (
        f"Delay-sum steering grid: {n_dirs} directions "
        f"({args.n_az}×{args.n_el}, upper hemisphere z≥0)"
    )
    ax.set_title(title, fontsize=11)

    summary = (
        f"Grid spacing (approx.): d_az ~ {az_step:.2f} deg, d_el ~ {el_step:.2f} deg "
        f"(el from 0 to {args.el_max:.1f} deg).\n"
        f"Peak pick is limited by this lattice (often ~half a cell or worse with noise).\n"
        f"Illustrative narrowband main-lobe scale at {f_mid:.0f} Hz, ring diameter {d_ap:.2f} m: "
        f"Rayleigh width ~{rayleigh_deg:.2f} deg (physics; broadband smears this)."
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=9, wrap=True)

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    print(summary.replace("\n", " "))
    print(f"Total steering vectors: {n_dirs}")

    if args.save:
        fig.savefig(args.save, dpi=160, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
