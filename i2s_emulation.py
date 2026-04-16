"""Map 8 microphones to 4 stereo I2S streams (interleaved L, R per frame)."""

from __future__ import annotations

import numpy as np

# Pair i: left = mic 2*i, right = mic 2*i + 1
N_PAIRS = 4
N_MICS = 8


def mics_to_i2s_interleaved(samples_per_mic: np.ndarray) -> np.ndarray:
    """
    Pack (n_mics, n_samples) into one interleaved stream.

    Order per time index: L0, R0, L1, R1, L2, R2, L3, R3 (mics 0..7).

    Returns shape (8 * n_samples,).
    """
    if samples_per_mic.shape[0] != N_MICS:
        raise ValueError(f"Expected {N_MICS} mics, got {samples_per_mic.shape[0]}")
    return np.ascontiguousarray(samples_per_mic.T).ravel()


def i2s_interleaved_to_mics(interleaved: np.ndarray, n_samples: int) -> np.ndarray:
    """Unpack interleaved stream to (8, n_samples)."""
    if interleaved.size != N_MICS * n_samples:
        raise ValueError("Length mismatch")
    block = interleaved.reshape(n_samples, N_MICS).T
    return block
