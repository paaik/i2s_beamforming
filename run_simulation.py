#!/usr/bin/env python3
"""
8-mic circular array simulation: drone acoustics, I2S-style packing, delay-sum scan + track.

Example:
  python run_simulation.py --path helix --duration 12 --save plot.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import beamformer
import drone_acoustic
import flight_paths
import geometry
import i2s_emulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="I2S 8-mic beamformer drone tracker simulation")
    p.add_argument(
        "--path",
        choices=sorted(flight_paths.PATH_REGISTRY.keys()),
        default="helix",
        help="Named flight path",
    )
    p.add_argument("--duration", type=float, default=10.0, help="Path duration (s)")
    p.add_argument("--fs-audio", type=float, default=48_000.0, help="Acoustic sample rate (Hz)")
    p.add_argument("--frame-rate", type=float, default=20.0, help="Tracker update rate (Hz)")
    p.add_argument("--chunk-ms", type=float, default=32.0, help="Audio chunk per tracker frame (ms)")
    p.add_argument("--n-az", type=int, default=36, help="Hemisphere scan azimuth bins")
    p.add_argument("--n-el", type=int, default=10, help="Hemisphere scan elevation bins (0..~89°)")
    p.add_argument("--smooth", type=float, default=0.4, help="Direction smoother alpha (0=no smooth, 1=only new)")
    p.add_argument("--snr-db", type=float, default=22.0, help="Mic noise SNR (dB); omit with negative to disable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="", help="Save figure to path (png) instead of anim")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def run_tracker_frames(
    path_xyz: np.ndarray,
    mic_xyz: np.ndarray,
    fs_audio: float,
    frame_rate: float,
    chunk_samples: int,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    dirs: np.ndarray,
    rng: np.random.Generator,
    smooth_alpha: float,
    snr_db: float | None,
) -> tuple[list, list, list, list]:
    """Returns lists: true_az, true_el, est_az, est_el per frame."""
    n_frames = path_xyz.shape[0]
    true_az, true_el = [], []
    est_az, est_el = [], []
    center = np.zeros(3)
    u_filt: np.ndarray | None = None

    for k in range(n_frames):
        src = path_xyz[k]
        u_true = geometry.direction_toward_source(src, center)
        az_t, el_t = geometry.cartesian_to_az_el_deg(u_true)
        true_az.append(az_t)
        true_el.append(el_t)

        drone = drone_acoustic.synthesize_drone_segment(chunk_samples, fs_audio, rng)
        delays, gains = drone_acoustic.geometric_delays_and_gains(src, mic_xyz)
        mics = drone_acoustic.mic_signals_from_source(drone, fs_audio, delays, gains, rng, snr_db=snr_db)

        # Optional: pack as 4x stereo I2S stream (validates channel order)
        _i2s = i2s_emulation.mics_to_i2s_interleaved(mics.astype(np.float32))
        _ = _i2s  # would be written to DMA / file in hardware

        pmap = beamformer.hemisphere_power_map(mics, fs_audio, mic_xyz, dirs)
        az_p, el_p, _ = beamformer.peak_direction(pmap, az_grid, el_grid)
        u_peak = geometry.unit_vector_from_angles(az_p, el_p)
        if u_filt is None:
            u_filt = u_peak.copy()
        else:
            u_filt = smooth_alpha * u_peak + (1.0 - smooth_alpha) * u_filt
            u_filt /= np.linalg.norm(u_filt) + 1e-15
        az_e, el_e = geometry.cartesian_to_az_el_deg(u_filt)
        est_az.append(az_e)
        est_el.append(el_e)

    return true_az, true_el, est_az, est_el


def plot_results(
    path_xyz: np.ndarray,
    true_az: list[float],
    true_el: list[float],
    est_az: list[float],
    est_el: list[float],
    mic_xyz: np.ndarray,
    title: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(11, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], "b-", lw=1.2, label="True path")
    ax3d.scatter(mic_xyz[:, 0], mic_xyz[:, 1], mic_xyz[:, 2], c="k", s=28, label="Mics")
    scale = 3.0
    n_est = len(est_az)
    steer_tips = np.zeros((n_est, 3), dtype=np.float64)
    for k in range(n_est):
        u = geometry.unit_vector_from_angles(est_az[k], est_el[k])
        steer_tips[k] = scale * u
    ax3d.plot(
        steer_tips[:, 0],
        steer_tips[:, 1],
        steer_tips[:, 2],
        color="green",
        ls="-",
        lw=1.3,
        alpha=0.9,
        label="Est. steer locus",
    )
    for k in range(0, n_est, max(1, n_est // 40)):
        ax3d.plot(
            [0, steer_tips[k, 0]],
            [0, steer_tips[k, 1]],
            [0, steer_tips[k, 2]],
            color="red",
            alpha=0.35,
            lw=0.8,
        )
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(loc="upper left", fontsize=8)
    lim = float(np.max(np.linalg.norm(path_xyz, axis=1)) * 1.15) + 1.0
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(0.0, max(lim, 2.0))

    axp = fig.add_subplot(1, 2, 2)
    t = np.arange(len(true_az), dtype=np.float64) / len(true_az)
    axp.plot(t, true_az, "b-", label="True az")
    axp.plot(t, est_az, "r--", label="Est az")
    axp.set_xlabel("Normalized time")
    axp.set_ylabel("Azimuth (deg)")
    axp.legend(loc="upper right", fontsize=8)
    axp2 = axp.twinx()
    axp2.plot(t, true_el, color="0.45", ls="-", label="True el")
    axp2.plot(t, est_el, color="orange", ls="--", label="Est el")
    axp2.set_ylabel("Elevation (deg)")
    lines1, lab1 = axp.get_legend_handles_labels()
    lines2, lab2 = axp2.get_legend_handles_labels()
    axp.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=7)
    axp.set_title("Bearing vs time")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    mic_xyz = geometry.mic_positions_circular_xy(8, 0.03)
    az_g, el_g, dirs = geometry.upper_hemisphere_grid(args.n_az, args.n_el)

    path_fn = flight_paths.PATH_REGISTRY[args.path]
    path_xyz = path_fn(args.duration, args.frame_rate)

    chunk_samples = max(256, int(args.fs_audio * args.chunk_ms / 1000.0))
    true_az, true_el, est_az, est_el = run_tracker_frames(
        path_xyz,
        mic_xyz,
        args.fs_audio,
        args.frame_rate,
        chunk_samples,
        az_g,
        el_g,
        dirs,
        rng,
        args.smooth,
        args.snr_db if args.snr_db >= 0 else None,
    )

    title = f"8-mic delay-sum tracker — path={args.path}, fs={args.fs_audio:.0f} Hz"
    fig = plot_results(path_xyz, true_az, true_el, est_az, est_el, mic_xyz, title)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")

    if not args.no_show and not args.save:
        plt.show()
    elif args.save and not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
