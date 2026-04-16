"""
Simulated I2S-style serialization: float waveforms ↔ quantized PCM ↔ serial bitstream.

This does not model BCLK/LRCLK timing; it encodes the same sample order as
``i2s_emulation.mics_to_i2s_interleaved``: per time index,
L0, R0, L1, R1, L2, R2, L3, R3 (8 values), each as *bits* MSB-first (24-bit two's complement).

Typical use: ``float_interleaved`` (length 8 * n_samples) → bytes or bits → decode → float.
"""

from __future__ import annotations

import numpy as np

from i2s_emulation import N_MICS

INT24_MAX = (1 << 23) - 1
INT24_MIN = -(1 << 23)


def normalize_peak(x: np.ndarray, peak: float = 0.99) -> tuple[np.ndarray, float]:
    """Scale so max |x| == peak (avoid clipping at ADC full scale). Returns (scaled, inv_scale)."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-15:
        return x.astype(np.float64), 1.0
    s = peak / m
    return (x.astype(np.float64) * s, s)


def float_to_int24(x: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and quantize to 24-bit signed integers."""
    x = np.clip(x, -1.0, 1.0)
    return np.rint(x * INT24_MAX).astype(np.int32).clip(INT24_MIN, INT24_MAX)


def int24_to_float(i: np.ndarray) -> np.ndarray:
    return i.astype(np.float64) / float(INT24_MAX)


def pack_int24_big_endian(i24: np.ndarray) -> bytes:
    """Flatten int32 (24-bit range) to 3-byte big-endian PCM octets."""
    v = i24.astype(np.int32)
    b0 = ((v >> 16) & 0xFF).astype(np.uint8)
    b1 = ((v >> 8) & 0xFF).astype(np.uint8)
    b2 = (v & 0xFF).astype(np.uint8)
    return np.stack([b0, b1, b2], axis=-1).reshape(-1).tobytes()


def unpack_int24_big_endian(data: bytes) -> np.ndarray:
    """PCM24 big-endian octets → int32 array (length len(data)//3)."""
    u8 = np.frombuffer(data, dtype=np.uint8)
    if u8.size % 3 != 0:
        raise ValueError("Byte length must be multiple of 3")
    u8 = u8.reshape(-1, 3)
    val = (
        u8[:, 0].astype(np.int32) << 16 | u8[:, 1].astype(np.int32) << 8 | u8[:, 2].astype(np.int32)
    )
    val = np.where(val >= 0x800000, val - 0x1000000, val)
    return val.astype(np.int32)


def interleaved_float_to_pcm24_bytes(
    interleaved_f32: np.ndarray,
    *,
    normalize_to_peak: float | None = 0.99,
) -> tuple[bytes, dict]:
    """
    ``interleaved_f32``: length ``8 * n_samples``, order L0,R0,...,L3,R3 per frame.

    Returns (pcm_bytes, meta) where meta has ``inv_scale`` if normalized.
    """
    x = np.asarray(interleaved_f32, dtype=np.float64).ravel()
    if x.size % N_MICS != 0:
        raise ValueError(f"Length must be multiple of {N_MICS}")
    meta: dict = {"normalize_to_peak": normalize_to_peak}
    inv_scale = 1.0
    if normalize_to_peak is not None:
        x, s = normalize_peak(x, normalize_to_peak)
        inv_scale = 1.0 / s if s > 0 else 1.0
        meta["scale_applied"] = s
        meta["inv_scale"] = inv_scale
    i24 = float_to_int24(x)
    return pack_int24_big_endian(i24), meta


def pcm24_bytes_to_interleaved_float(data: bytes, *, inv_scale: float = 1.0) -> np.ndarray:
    """Inverse of ``interleaved_float_to_pcm24_bytes`` (apply inv_scale to recover level)."""
    i24 = unpack_int24_big_endian(data)
    f = int24_to_float(i24)
    return (f * inv_scale).astype(np.float32)


def pcm24_bytes_to_serial_bits(data: bytes) -> np.ndarray:
    """
    Expand packed PCM octets to a 1D array of bits {0,1}, MSB first within each byte
    (same order as would be shifted out on an I2S data line byte-serialized).
    """
    u8 = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(u8).astype(np.uint8)


def serial_bits_to_pcm24_bytes(bits: np.ndarray) -> bytes:
    """Inverse of ``pcm24_bytes_to_serial_bits``; ``bits`` length must be multiple of 8."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if b.size % 8 != 0:
        raise ValueError("Bit length must be multiple of 8")
    if np.any((b != 0) & (b != 1)):
        raise ValueError("Bits must be 0 or 1")
    u8 = np.packbits(b)
    return u8.tobytes()


def interleaved_float_to_serial_bitstream(
    interleaved_f32: np.ndarray,
    *,
    bits_per_sample: int = 24,
    normalize_to_peak: float | None = 0.99,
) -> tuple[np.ndarray, dict]:
    """
    Build a logical serial stream: for each time index, channel order L0..R3, each channel
    ``bits_per_sample`` bits MSB-first (two's complement).

    Returns (bit_array uint8 0/1, meta). Length = n_frames * 8 * bits_per_sample.
    """
    if bits_per_sample != 24:
        raise NotImplementedError("Only 24-bit mode implemented")
    x = np.asarray(interleaved_f32, dtype=np.float64).ravel()
    if x.size % N_MICS != 0:
        raise ValueError(f"Length must be multiple of {N_MICS}")
    n = x.size // N_MICS
    meta: dict = {"bits_per_sample": bits_per_sample, "n_frames": n}
    inv_scale = 1.0
    if normalize_to_peak is not None:
        x, s = normalize_peak(x, normalize_to_peak)
        inv_scale = 1.0 / s if s > 0 else 1.0
        meta["scale_applied"] = s
        meta["inv_scale"] = inv_scale
    x = x.reshape(n, N_MICS)
    iv = float_to_int24(x)
    out = np.zeros((n, N_MICS, bits_per_sample), dtype=np.uint8)
    for b in range(bits_per_sample):
        shift = bits_per_sample - 1 - b
        out[:, :, b] = (iv >> shift) & 1
    bits = out.reshape(-1)
    return bits, meta


def serial_bitstream_to_interleaved_float(
    bits: np.ndarray,
    *,
    bits_per_sample: int = 24,
    inv_scale: float = 1.0,
) -> np.ndarray:
    """Decode bitstream from ``interleaved_float_to_serial_bitstream``."""
    if bits_per_sample != 24:
        raise NotImplementedError("Only 24-bit mode implemented")
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if b.size % (N_MICS * bits_per_sample) != 0:
        raise ValueError("Bit length must be n_frames * 8 * 24")
    n = b.size // (N_MICS * bits_per_sample)
    pack = b.reshape(n, N_MICS, bits_per_sample)
    iv = np.zeros((n, N_MICS), dtype=np.int32)
    for k in range(bits_per_sample):
        shift = bits_per_sample - 1 - k
        iv |= pack[:, :, k].astype(np.int32) << shift
    # sign-extend 24-bit
    iv = np.where(iv >= 0x800000, iv - 0x1000000, iv)
    iv = iv.clip(INT24_MIN, INT24_MAX)
    f = int24_to_float(iv).ravel() * inv_scale
    return f.astype(np.float32)


def roundtrip_bytes(interleaved_f32: np.ndarray, **kw) -> np.ndarray:
    data, meta = interleaved_float_to_pcm24_bytes(interleaved_f32, **kw)
    return pcm24_bytes_to_interleaved_float(data, inv_scale=meta.get("inv_scale", 1.0))


def roundtrip_bits(interleaved_f32: np.ndarray, **kw) -> np.ndarray:
    bits, meta = interleaved_float_to_serial_bitstream(interleaved_f32, **kw)
    return serial_bitstream_to_interleaved_float(bits, inv_scale=meta.get("inv_scale", 1.0))
