"""Circular 8-microphone array geometry and direction vectors (upper hemisphere)."""

from __future__ import annotations

import numpy as np

C_SOUND = 343.0  # m/s, approximate at 20 °C


def mic_positions_circular_xy(
    n_mics: int = 8,
    radius_m: float = 0.03,
    *,
    start_deg: float = 0.0,
) -> np.ndarray:
    """
    Mic positions in the z=0 plane, counterclockwise from +x when viewed from +z.
    Shape (n_mics, 3).
    """
    angles = np.deg2rad(start_deg + np.arange(n_mics, dtype=np.float64) * (360.0 / n_mics))
    x = radius_m * np.cos(angles)
    y = radius_m * np.sin(angles)
    z = np.zeros_like(x)
    return np.column_stack([x, y, z])


def unit_vector_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Unit vector pointing from array origin toward a direction in the upper hemisphere.

    Convention: azimuth 0 = +x, 90° = +y; elevation 0 = horizon (xy plane), 90° = +z (zenith).
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    ce = np.cos(el)
    return np.array([ce * np.cos(az), ce * np.sin(az), np.sin(el)], dtype=np.float64)


def upper_hemisphere_grid(
    n_az: int,
    n_el: int,
    el_max_deg: float = 89.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regular grid of (az, el) in degrees covering full azimuth and elevation [0, el_max].

    Returns
    -------
    az_grid, el_grid : 2D arrays (n_el, n_az) in degrees
    dirs : (n_el, n_az, 3) unit vectors
    """
    az = np.linspace(0.0, 360.0, n_az, endpoint=False)
    el = np.linspace(0.0, el_max_deg, n_el)
    az_g, el_g = np.meshgrid(az, el, indexing="xy")
    dirs = np.stack(
        [
            np.cos(np.deg2rad(el_g)) * np.cos(np.deg2rad(az_g)),
            np.cos(np.deg2rad(el_g)) * np.sin(np.deg2rad(az_g)),
            np.sin(np.deg2rad(el_g)),
        ],
        axis=-1,
    )
    return az_g, el_g, dirs


def cartesian_to_az_el_deg(u: np.ndarray) -> tuple[float, float]:
    """Unit or arbitrary vector u -> (azimuth_deg, elevation_deg), elevation in [0, 90] for u_z >= 0."""
    u = np.asarray(u, dtype=np.float64).reshape(3)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return 0.0, 0.0
    u = u / n
    xy = np.hypot(u[0], u[1])
    el = float(np.degrees(np.arctan2(u[2], xy)))
    az = float(np.degrees(np.arctan2(u[1], u[0])))
    if az < 0:
        az += 360.0
    return az, el


def direction_toward_source(source_xyz: np.ndarray, array_center: np.ndarray) -> np.ndarray:
    """Unit vector from array_center toward source_xyz."""
    v = source_xyz - array_center
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return v / n
