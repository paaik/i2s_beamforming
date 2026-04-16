#!/usr/bin/env python3
"""
Build simulated 8-mic drone audio, pack as I2S-style PCM / serial bits, decode back to float.

Example:
  python demo_i2s_extract.py --samples 4800 --seed 1
  python demo_i2s_extract.py --write-pcm out.pcm24 --write-bits-npy out_bits.npy
"""

from __future__ import annotations

import argparse

import numpy as np

import drone_acoustic
import geometry
import i2s_bitstream
import i2s_emulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulated I2S bitstream ↔ float (drone + 8 mics)")
    p.add_argument("--fs", type=float, default=48_000.0)
    p.add_argument("--samples", type=int, default=4096, help="Samples per mic channel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--source",
        type=float,
        nargs=3,
        default=[6.0, 2.0, 4.0],
        metavar=("X", "Y", "Z"),
        help="Drone position (m) for propagation",
    )
    p.add_argument("--snr-db", type=float, default=-1.0, help="Negative disables noise")
    p.add_argument("--write-pcm", type=str, default="", help="Write raw big-endian PCM24 octets")
    p.add_argument("--write-bits-npy", type=str, default="", help="Save serial 0/1 bit array .npy")
    p.add_argument("--no-normalize", action="store_true", help="Map float [-1,1] without peak scaling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    mic_xyz = geometry.mic_positions_circular_xy(8, 0.03)
    src = np.asarray(args.source, dtype=np.float64)

    drone = drone_acoustic.synthesize_drone_segment(args.samples, args.fs, rng)
    delays, gains = drone_acoustic.geometric_delays_and_gains(src, mic_xyz)
    snr = None if args.snr_db < 0 else args.snr_db
    mics = drone_acoustic.mic_signals_from_source(drone, args.fs, delays, gains, rng, snr_db=snr)

    interleaved = i2s_emulation.mics_to_i2s_interleaved(mics.astype(np.float32))
    norm = None if args.no_normalize else 0.99

    pcm_bytes, meta_pcm = i2s_bitstream.interleaved_float_to_pcm24_bytes(
        interleaved, normalize_to_peak=norm
    )
    bits, meta_bits = i2s_bitstream.interleaved_float_to_serial_bitstream(
        interleaved, normalize_to_peak=norm
    )

    recon_pcm = i2s_bitstream.pcm24_bytes_to_interleaved_float(
        pcm_bytes, inv_scale=meta_pcm.get("inv_scale", 1.0)
    )
    recon_bits = i2s_bitstream.serial_bitstream_to_interleaved_float(
        bits, inv_scale=meta_bits.get("inv_scale", 1.0)
    )

    i64 = interleaved.astype(np.float64)
    # Decoder applies inv_scale so waveform matches original float levels before quantize
    err_pcm = float(np.sqrt(np.mean((recon_pcm.astype(np.float64) - i64) ** 2)))
    err_bits = float(np.sqrt(np.mean((recon_bits.astype(np.float64) - i64) ** 2)))

    print("Simulated 8-mic drone chunk:")
    print(f"  fs={args.fs:.0f} Hz, samples={args.samples}, source={tuple(float(x) for x in src)}")
    print(f"  PCM24 bytes: {len(pcm_bytes)} (= {args.samples} * 8 ch * 3)")
    print(f"  Serial bits (per-sample MSB-first, L0..R3): {bits.size}")
    print(f"  Meta (PCM): {meta_pcm}")
    print(f"  RMS float error after PCM24 roundtrip (vs scaled input): {err_pcm:.3e}")
    print(f"  RMS float error after serial-bit roundtrip: {err_bits:.3e}")

    if args.write_pcm:
        with open(args.write_pcm, "wb") as f:
            f.write(pcm_bytes)
        print(f"  Wrote {args.write_pcm}")
    if args.write_bits_npy:
        np.save(args.write_bits_npy, bits)
        print(f"  Wrote {args.write_bits_npy}")


if __name__ == "__main__":
    main()
