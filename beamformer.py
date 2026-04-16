"""
Delay-and-sum beamformer; upper-hemisphere power map and peak picking.

Steering delays match ``mic_signals_from_source`` (spherical delay s(t − d_i/c)):
τ_i = −(r_i·û)/c after mean removal. Synthetic plane waves written as s(t − (û·r_i)/c)
use the opposite phase slope and would need τ_i = +(r_i·û)/c instead.
"""

from __future__ import annotations

import numpy as np

from geometry import C_SOUND


def _advance_channel(x: np.ndarray, advance_samples: float) -> np.ndarray:
    """Fractional advance: output[k] ≈ input[k + advance_samples] (plane-wave alignment)."""
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64)
    return np.interp(t + advance_samples, t, x, left=0.0, right=0.0)


def delay_sum_output(
    mic_sig: np.ndarray,
    fs: float,
    mic_xyz: np.ndarray,
    steer_u: np.ndarray,
) -> np.ndarray:
    """
    Time-domain delay-and-sum steered toward unit vector steer_u (array toward source).

    mic_sig: (n_mics, n_samples)
    Returns 1D beamformer output (n_samples,).

    Steering uses τ_i = −(r_i·û)/c (mean-centered) so phase slope matches finite-range
    propagation modeled as s(t − |p−r_i|/c) (see module doc / project notes).
    """
    steer_u = np.asarray(steer_u, dtype=np.float64)
    steer_u = steer_u / (np.linalg.norm(steer_u) + 1e-15)
    n_mics = mic_sig.shape[0]
    tau = -(mic_xyz @ steer_u) / C_SOUND
    tau = tau - np.mean(tau)
    aligned = np.zeros_like(mic_sig)
    for i in range(n_mics):
        aligned[i] = _advance_channel(mic_sig[i], tau[i] * fs)
    return np.mean(aligned, axis=0)


def delay_sum_rms_power(
    mic_sig: np.ndarray,
    fs: float,
    mic_xyz: np.ndarray,
    steer_u: np.ndarray,
) -> float:
    y = delay_sum_output(mic_sig, fs, mic_xyz, steer_u)
    return float(np.sqrt(np.mean(y**2) + 1e-20))


def hemisphere_power_map(
    mic_sig: np.ndarray,
    fs: float,
    mic_xyz: np.ndarray,
    dirs: np.ndarray,
) -> np.ndarray:
    """
    dirs: (n_el, n_az, 3) unit vectors. Returns power map (n_el, n_az).

    FFT-based delay-and-sum: Y(ω) = Σ_i X_i(ω) exp(j ω τ_i), power ∝ mean_f |Y|².
    τ_i = −(r_i·û)/c (mean-centered), consistent with spherical delay synthesis in
    ``drone_acoustic.mic_signals_from_source``.
    """
    _, n = mic_sig.shape
    x = np.fft.rfft(mic_sig, axis=1)
    f_hz = np.fft.rfftfreq(n, d=1.0 / fs)
    omega = (2 * np.pi * f_hz).astype(np.float64)
    ne, na, _ = dirs.shape
    dirs_f = dirs.reshape(-1, 3)
    tau = -(mic_xyz @ dirs_f.T) / C_SOUND
    tau -= np.mean(tau, axis=0, keepdims=True)
    steer = np.exp(1j * omega[:, np.newaxis, np.newaxis] * tau[np.newaxis, :, :])
    spec = np.swapaxes(x, 0, 1)[:, :, np.newaxis]
    y = np.sum(spec * steer, axis=1)
    power = np.mean(np.abs(y) ** 2, axis=0)
    return power.reshape(ne, na)


def peak_direction(
    power: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
) -> tuple[float, float, tuple[int, int]]:
    """Returns az_deg, el_deg, (i_el, i_az) index of maximum."""
    idx = np.unravel_index(int(np.argmax(power)), power.shape)
    return float(az_grid[idx]), float(el_grid[idx]), (int(idx[0]), int(idx[1]))
