"""Parametric 3D flight paths for the virtual drone (array at origin, +z up)."""

from __future__ import annotations

import numpy as np

from geometry import unit_vector_from_angles


def _resample_polyline(waypoints_xyz: np.ndarray, n_out: int) -> np.ndarray:
    """Evenly spaced samples along piecewise-linear segments (arc-length parameter)."""
    wp = np.asarray(waypoints_xyz, dtype=np.float64)
    if wp.shape[0] < 2:
        return np.repeat(wp, max(1, n_out), axis=0)[:n_out]
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-12:
        return np.repeat(wp[:1], n_out, axis=0)
    tgt = np.linspace(0.0, total, n_out)
    out = np.zeros((n_out, 3), dtype=np.float64)
    for i, s in enumerate(tgt):
        j = int(np.searchsorted(cum, s, side="right")) - 1
        j = max(0, min(j, len(seg) - 1))
        denom = seg[j] + 1e-15
        u = (s - cum[j]) / denom
        out[i] = wp[j] + u * (wp[j + 1] - wp[j])
    return out


def path_straight_overhead(duration_s: float, fs_frame: float) -> np.ndarray:
    """Fly along +y at fixed height, crossing over the array."""
    n = max(2, int(duration_s * fs_frame))
    t = np.linspace(0.0, 1.0, n)
    # Start in front (-y), pass over origin, exit +y
    y = -8.0 + t * 16.0
    x = np.zeros_like(y)
    z = np.full_like(y, 4.0)
    return np.column_stack([x, y, z])


def path_circular_arc(duration_s: float, fs_frame: float) -> np.ndarray:
    """Arc in a vertical plane (xz), radius ~6 m, center offset in x."""
    n = max(2, int(duration_s * fs_frame))
    theta = np.linspace(0.25 * np.pi, 0.85 * np.pi, n)
    r = 7.0
    cx, cz = 2.0, -2.0
    x = cx + r * np.cos(theta)
    z = cz + r * np.sin(theta)
    z = np.maximum(z, 1.5)
    y = np.linspace(-3.0, 3.0, n)
    return np.column_stack([x, y, z])


def path_helix(duration_s: float, fs_frame: float) -> np.ndarray:
    """Ascending helix around +z axis."""
    n = max(2, int(duration_s * fs_frame))
    t = np.linspace(0.0, duration_s, n)
    omega = 0.9
    r = 5.0
    x = r * np.cos(omega * t)
    y = r * np.sin(omega * t)
    z = 2.0 + 0.35 * t
    return np.column_stack([x, y, z])


def path_approach_hover_leave(duration_s: float, fs_frame: float) -> np.ndarray:
    """Approach from far +x, loiter near zenith, depart -x."""
    n = max(2, int(duration_s * fs_frame))
    t = np.linspace(0.0, 1.0, n)
    # piecewise in t
    p = np.zeros((n, 3))
    for i, u in enumerate(t):
        if u < 0.35:
            v = u / 0.35
            p[i] = np.array([12.0 * (1.0 - v), 0.0, 1.5 + 5.0 * v])
        elif u < 0.65:
            v = (u - 0.35) / 0.3
            ang = v * 2 * np.pi
            p[i] = np.array([1.5 * np.cos(ang), 1.5 * np.sin(ang), 6.0])
        else:
            v = (u - 0.65) / 0.35
            p[i] = np.array([-6.0 * v, -3.0 * v, 6.0 * (1.0 - 0.4 * v)])
    return p


def path_figure_eight(duration_s: float, fs_frame: float) -> np.ndarray:
    """Lemniscate-like curve in horizontal plane with mild altitude wobble."""
    n = max(2, int(duration_s * fs_frame))
    s = np.linspace(0.0, 2 * np.pi, n)
    scale = 6.0
    x = scale * np.sin(s)
    y = scale * np.sin(s) * np.cos(s)
    z = 4.0 + 0.8 * np.sin(2 * s)
    return np.column_stack([x, y, z])


def path_left_to_right_5m(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    Straight line along +x (left → right in a typical x-horizontal plot), y = 0,
    constant height z = 5 m above the array (mics in z = 0 plane).
    """
    n = max(2, int(duration_s * fs_frame))
    t = np.linspace(0.0, 1.0, n)
    x = -12.0 + t * 24.0
    y = np.zeros_like(x)
    z = np.full_like(x, 5.0)
    return np.column_stack([x, y, z])


def path_left_to_right_5m_side_3m(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    Same as ``path_left_to_right_5m`` (along +x, z = 5 m) but offset **3 m** in +y so the
    track stays parallel to the array x-axis and passes **3 m to the side** (horizontal CPA).
    """
    n = max(2, int(duration_s * fs_frame))
    t = np.linspace(0.0, 1.0, n)
    x = -12.0 + t * 24.0
    y = np.full_like(x, 3.0)
    z = np.full_like(x, 5.0)
    return np.column_stack([x, y, z])


def path_circle_10m_diameter_1m_altitude(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    Horizontal circle centered on the array (xy), z = 1 m above the mic plane,
    diameter 10 m (radius 5 m). One full orbit over the path duration.
    """
    n = max(2, int(duration_s * fs_frame))
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = 5.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, 1.0)
    return np.column_stack([x, y, z])


def path_parabolic_flyby(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    Parabolic flyby in the x–z plane (y = 0): straight-line motion in x, height z = a·x² + b
    so altitude is lowest over the array at x = 0 (closest point of approach), higher at the ends.
    """
    n = max(2, int(duration_s * fs_frame))
    t = np.linspace(0.0, 1.0, n)
    x_half = 14.0
    x = -x_half + t * (2.0 * x_half)
    y = np.zeros_like(x)
    z_cpa = 2.8
    z_ends = 9.0
    z = z_cpa + (z_ends - z_cpa) * (x / x_half) ** 2
    return np.column_stack([x, y, z])


def _smooth_moving_average_1d(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or a.size < window:
        return a
    w = window | 1
    pad = w // 2
    ap = np.pad(a.astype(np.float64), (pad, pad), mode="edge")
    k = np.ones(w, dtype=np.float64) / float(w)
    out = np.convolve(ap, k, mode="valid")
    return out.astype(np.float64)


def path_organic_orbit(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    Smooth loop around the array with gentle radius/heading drift and wide elevation range.
    """
    n = max(2, int(duration_s * fs_frame))
    u = np.linspace(0.0, 1.0, n)
    th = 2.0 * np.pi * u * (1.02 + 0.03 * np.cos(np.pi * u))
    r0 = 5.4
    r = r0 + 0.48 * np.sin(1.05 * th + 0.25)
    r = np.maximum(r, 3.4)
    psi = 0.14 * np.sin(0.85 * th + 0.35) + 0.05 * np.sin(1.7 * th + 0.9)
    ang = th + psi
    x = r * np.cos(ang)
    y = r * np.sin(ang) * 0.96
    z = 3.85 + 1.72 * np.sin(1.0 * th + 0.45) + 0.92 * np.sin(1.95 * th + 1.05)
    z = np.maximum(z, 0.95)
    p = np.column_stack([x, y, z])
    win = max(3, min(15, n // 40))
    for j in range(3):
        p[:, j] = _smooth_moving_average_1d(p[:, j], win)
    return p


def path_spiral_cone_in(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    **Curvy** tightening spiral: several **circular** orbits while horizontal radius **shrinks**.
    **z** runs from **2 m** at the start to **10 m** at the end, with **dynamic** wobble that
    vanishes at both endpoints (``sin(π t)`` envelope) so altitude stays bounded between ~2–10 m.
    """
    n = max(2, int(duration_s * fs_frame))
    t = np.linspace(0.0, 1.0, n)
    n_turns = 4.25
    theta = 2.0 * np.pi * n_turns * t
    # Horizontal cone: large radius early → small radius late (footprint tightens)
    r_outer = 7.2
    r_inner = 1.05
    fade = np.power(np.maximum(1.0 - t, 0.0), 1.12)
    r = r_inner + (r_outer - r_inner) * fade
    r *= 1.0 + 0.065 * np.sin(2.65 * theta + 0.35)
    r = np.maximum(r, 0.82)
    # Slight phase wobble so the loop is not a perfect circle
    phase = 0.16 * np.sin(1.25 * theta)
    x = r * np.cos(theta + phase)
    y = r * np.sin(theta + phase)
    # z: 2 m → 10 m baseline; wobble strongest mid-path, zero at t=0 and t=1
    z0, z1 = 2.0, 10.0
    z_base = z0 + (z1 - z0) * t
    env = np.sin(np.pi * t)
    z = z_base + env * (
        fade * (1.25 * np.sin(2.9 * theta + 0.4) + 0.52 * np.sin(5.6 * theta + 0.9))
        + 0.38 * (1.0 - fade) * np.sin(7.2 * theta)
    )
    z = np.clip(z, z0, z1)
    p = np.column_stack([x, y, z])
    win = max(3, min(13, n // 45))
    for j in range(3):
        p[:, j] = _smooth_moving_average_1d(p[:, j], win)
    p[0, 2] = z0
    p[-1, 2] = z1
    return p


def path_beamformer_precision_extent(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    **Geometric** path on a sphere of radius **R** m: waypoints lie on an **(azimuth × elevation)**
    lattice whose steps match the **default** hemisphere scan in ``run_simulation.py``
    (**``--n-az 36``**, **``--n-el 10``**, **``el_max 89°``**): **Δaz = 10°**, **Δel ≈ 9.89°**.
    The track is **piecewise linear** (straight chords in 3D) between neighbors in a **serpentine**
    order—this highlights the **angular cell size** that limits peak-picking **precision** for
    that grid (bearing estimates snap to ~one cell unless refined).

    The lattice is placed in the **opposite** sector from the original low-az / low-el corner:
    **azimuth shifted by 180°** and **higher elevations** so the pattern sits in the far upper
    hemisphere “corner” of the scan volume.
    """
    R = 6.0
    n_az, n_el, el_max = 36, 10, 89.0
    d_az = 360.0 / float(n_az)
    d_el = el_max / float(max(1, n_el - 1))
    # Was (32°, 12°); opposite corner: +180° az, upper el band (still 4×3 cells, inside 0–89° el)
    az0, el0 = 212.0, 56.5
    az_list = [az0 + i * d_az for i in range(4)]
    el_list = [el0 + j * d_el for j in range(3)]
    pts: list[np.ndarray] = []
    for j, el in enumerate(el_list):
        row = az_list if j % 2 == 0 else list(reversed(az_list))
        for az in row:
            u = unit_vector_from_angles(float(az), float(el))
            pts.append(R * u)
    wp = np.stack(pts, axis=0)
    n = max(2, int(duration_s * fs_frame))
    return _resample_polyline(wp, n)


def path_flyby_xneg5_y_span_z_sine(duration_s: float, fs_frame: float) -> np.ndarray:
    """
    **U-shaped** flyby in **x–y** (opening toward **+y**): **(−5, 5) → (−5, −5) → (5, −5) → (5, 5)** —
    down the left edge, **10 m** along the bottom, up the right edge (**10 m** per segment,
    **30 m** total). **z** is one continuous sinusoid (**1 m** amplitude about **5 m**, **3** cycles
    over the full path vs **s ∈ [0, 1]**).
    """
    n = max(2, int(duration_s * fs_frame))
    s = np.linspace(0.0, 1.0, n)
    third = 1.0 / 3.0
    x = np.where(
        s <= third,
        -5.0,
        np.where(s <= 2.0 * third, -5.0 + (s - third) / third * 10.0, 5.0),
    )
    y = np.where(
        s <= third,
        5.0 - (s / third) * 10.0,
        np.where(s <= 2.0 * third, -5.0, -5.0 + (s - 2.0 * third) / third * 10.0),
    )
    z_mean = 5.0
    amp = 1.0
    n_cycles = 3.0
    z = z_mean + amp * np.sin(2.0 * np.pi * n_cycles * s)
    return np.column_stack([x, y, z])


PATH_REGISTRY = {
    "straight": path_straight_overhead,
    "arc": path_circular_arc,
    "helix": path_helix,
    "approach": path_approach_hover_leave,
    "figure8": path_figure_eight,
    "left_right_5m": path_left_to_right_5m,
    "left_right_5m_side3m": path_left_to_right_5m_side_3m,
    "circle_10m_1m": path_circle_10m_diameter_1m_altitude,
    "parabolic_flyby": path_parabolic_flyby,
    "organic": path_organic_orbit,
    "spiral_cone_in": path_spiral_cone_in,
    "precision_extent": path_beamformer_precision_extent,
    "flyby_xneg5_y_sine": path_flyby_xneg5_y_span_z_sine,
    "flyby_x10_y_sine": path_flyby_xneg5_y_span_z_sine,  # alias; path is at x = −5 m
}
