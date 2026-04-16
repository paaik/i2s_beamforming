"""Virtual drone: broadband noise + motor harmonics, received at microphones."""

from __future__ import annotations

import numpy as np
from scipy import signal

from geometry import C_SOUND


def synthesize_drone_segment(
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    *,
    fundamental_hz: float = 120.0,
    n_harmonics: int = 8,
    harmonic_decay: float = 0.85,
    broadband_hp_hz: float = 80.0,
    broadband_lp_hz: float = 4000.0,
    harmonic_jitter_hz: float = 2.0,
) -> np.ndarray:
    """
    One segment of drone-like pressure waveform (arbitrary units, ~ unit variance).

    Motor: sum of harmonics with slow random FM.
    Body/airframe: band-limited noise.
    """
    t = np.arange(n_samples, dtype=np.float64) / fs
    x = np.zeros(n_samples, dtype=np.float64)

    # Broadband (colored noise)
    w = rng.standard_normal(n_samples)
    sos_hp = signal.butter(4, broadband_hp_hz, btype="high", fs=fs, output="sos")
    sos_lp = signal.butter(4, broadband_lp_hz, btype="low", fs=fs, output="sos")
    bb = signal.sosfilt(sos_hp, w)
    bb = signal.sosfilt(sos_lp, bb)
    bb /= np.std(bb) + 1e-12
    x += 0.55 * bb

    # Motor harmonics (slight per-segment detuning)
    for k in range(1, n_harmonics + 1):
        fk = k * fundamental_hz + rng.uniform(-harmonic_jitter_hz, harmonic_jitter_hz)
        amp = (harmonic_decay ** (k - 1)) / k
        x += amp * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))

    x -= np.mean(x)
    x /= np.std(x) + 1e-12
    return x


def geometric_delays_and_gains(
    source_xyz: np.ndarray,
    mic_xyz: np.ndarray,
    *,
    min_dist_m: float = 0.15,
    rolloff_exp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Absolute propagation delay (seconds) and amplitude gain per microphone.

    delay[i] = dist_i / c; received signal ~ s(t - delay[i]) * gain[i].
    gain uses reference distance min_dist_m for numerical stability.
    """
    d = np.linalg.norm(mic_xyz - source_xyz, axis=1)
    d = np.maximum(d, min_dist_m)
    delays = d / C_SOUND
    d0 = np.min(d)
    gains = (d0 / d) ** rolloff_exp
    return delays, gains


def mic_signals_from_source(
    source_waveform: np.ndarray,
    fs: float,
    delays_s: np.ndarray,
    gains: np.ndarray,
    rng: np.random.Generator,
    snr_db: float | None = 25.0,
) -> np.ndarray:
    """
    Apply fractional delays via linear interpolation; shape (n_mics, n_samples).

    delays_s[i]: absolute delay from source to mic i (seconds).
    """
    n_mics = len(delays_s)
    n = source_waveform.shape[0]
    mic_sig = np.zeros((n_mics, n), dtype=np.float64)
    t_idx = np.arange(n, dtype=np.float64)
    for i in range(n_mics):
        shift = delays_s[i] * fs
        mic_sig[i] = np.interp(t_idx - shift, t_idx, source_waveform) * gains[i]

    if snr_db is not None:
        noise = rng.standard_normal(mic_sig.shape)
        noise /= np.std(noise) + 1e-12
        sig_pow = np.mean(mic_sig**2) + 1e-12
        noise_pow = sig_pow / (10 ** (snr_db / 10.0))
        mic_sig += noise * np.sqrt(noise_pow)

    return mic_sig
