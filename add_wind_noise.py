#!/usr/bin/env python3
"""
add_wind_noise.py

Utility to add wind noise to clean speech WAV files.

Features:
- Add a wind noise segment (randomly cropped) to a clean file at the requested SNR (dB).
- Works on a single file pair or on a directory of clean files (picks random wind file for each clean file).
- Resamples audio to a target sample rate if requested.
- Converts stereo to mono by averaging channels.

Usage examples:

Single pair:
    python add_wind_noise.py --clean clean.wav --wind wind.wav --out noisy.wav --snr 0

Directory mode (apply wind to all WAVs in clean-dir):
    python add_wind_noise.py --clean-dir ./clean/ --wind-dir ./wind/ --out-dir ./noisy/ --snr 5

Options:
    --pad-noise: if wind noise is shorter than clean, tile it to fit instead of padding with zeros
    --sr: target sample rate (default: preserves each file's sample rate)
    --seed: random seed for reproducibility

Dependencies: soundfile, numpy. Optional: librosa or scipy for resampling.

"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import soundfile as sf

# optional resamplers
_have_librosa = False
_have_scipy = False
try:
    import librosa
    _have_librosa = True
except Exception:
    try:
        from scipy.signal import resample_poly
        _have_scipy = True
    except Exception:
        pass


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        # assume (n_samples, channels)
        if y.shape[0] < y.shape[1]:
            y = y.T
        return y.mean(axis=1)
    raise ValueError(f"Unsupported audio shape: {y.shape}")


def resample_if_needed(y: np.ndarray, orig_sr: int, target_sr: Optional[int]) -> (np.ndarray, int):
    if target_sr is None or orig_sr == target_sr:
        return y, orig_sr
    if _have_librosa:
        y_rs = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        return y_rs, target_sr
    if _have_scipy:
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        y_rs = resample_poly(y, up, down)
        return y_rs.astype(y.dtype), target_sr
    raise RuntimeError("No resampler available: install librosa (`pip install librosa`) or scipy (`pip install scipy`).")


def read_audio(path: str, target_sr: Optional[int] = None) -> (np.ndarray, int):
    data, sr = sf.read(path, dtype="float32")
    data = to_mono(np.asarray(data))
    data, sr = resample_if_needed(data, sr, target_sr)
    return data.astype(np.float32), sr


def crop_or_tile_noise(noise: np.ndarray, length: int, pad_mode: str = "tile") -> np.ndarray:
    # If noise is shorter than length, either tile or pad with zeros depending on pad_mode
    if noise.shape[0] >= length:
        # pick random start
        start = random.randint(0, noise.shape[0] - length)
        return noise[start:start + length]
    # shorter
    if pad_mode == "tile":
        # repeat the noise to reach required length and then crop
        reps = int(np.ceil(length / noise.shape[0]))
        tiled = np.tile(noise, reps)[:length]
        return tiled
    else:
        # zero pad
        return np.pad(noise, (0, length - noise.shape[0]), mode="constant")


def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    # Compute powers (mean squared)
    eps = 1e-10
    clean_power = np.mean(clean.astype(np.float64) ** 2) + eps
    noise_power = np.mean(noise.astype(np.float64) ** 2) + eps
    # desired noise power
    desired_noise_power = clean_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(desired_noise_power / noise_power)
    return noise * scale


def mix_with_wind(clean_path: str, wind_path: str, out_path: str, snr_db: float, target_sr: Optional[int] = None, pad_noise: bool = True):
    clean, sr_clean = read_audio(clean_path, target_sr=target_sr)
    wind, sr_wind = read_audio(wind_path, target_sr=sr_clean)

    # Ensure same sample rate
    if sr_wind != sr_clean:
        # should not happen because we resampled wind to sr_clean above, but just in case
        wind, sr_wind = resample_if_needed(wind, sr_wind, sr_clean)

    # Crop or tile wind to match clean length
    wind_seg = crop_or_tile_noise(wind, len(clean), pad_mode="tile" if pad_noise else "pad")

    scaled_noise = scale_noise_to_snr(clean, wind_seg, snr_db)
    noisy = clean + scaled_noise

    # Optional clipping to [-1, 1] to avoid overflow when saving as float
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / (max_val + 1e-8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, noisy.astype(np.float32), sr_clean)
    return out_path


def find_wind_for_each_clean(clean_files, wind_files):
    # for each clean file pick a random wind file (could be improved to pick different segments)
    for cf in clean_files:
        wf = random.choice(wind_files)
        yield cf, wf


def main():
    parser = argparse.ArgumentParser(description="Add wind noise to clean audio files at a desired SNR.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clean", type=str, help="Path to clean wav file")
    group.add_argument("--clean-dir", type=str, help="Directory with clean wav files")

    parser.add_argument("--wind", type=str, help="Path to wind noise wav file (or directory via --wind-dir)")
    parser.add_argument("--wind-dir", type=str, help="Directory with wind noise wav files")

    parser.add_argument("--out", type=str, help="Output path for single file mix")
    parser.add_argument("--out-dir", type=str, help="Output directory for directory mode")

    parser.add_argument("--snr", type=float, default=0.0, help="Target SNR in dB (clean relative to noise). Default 0 dB")
    parser.add_argument("--sr", type=int, default=None, help="Target sample rate for resampling (default: keep original)")
    parser.add_argument("--pad-noise", action="store_true", help="Tile noise to fit clean file if wind is shorter (default True in code)")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Gather wind files
    wind_files = []
    if args.wind_dir:
        wind_files = [os.path.join(args.wind_dir, f) for f in os.listdir(args.wind_dir) if f.lower().endswith('.wav')]
    if args.wind:
        if os.path.isdir(args.wind):
            wind_files.extend([os.path.join(args.wind, f) for f in os.listdir(args.wind) if f.lower().endswith('.wav')])
        else:
            wind_files.append(args.wind)
    wind_files = [w for w in wind_files if os.path.isfile(w)]
    if not wind_files:
        raise SystemExit("No wind noise files found. Provide --wind or --wind-dir with wav files.")

    # Single file mode
    if args.clean:
        if args.out is None:
            raise SystemExit("--out must be provided when using --clean")
        wind_choice = random.choice(wind_files)
        print(f"Mixing {args.clean} with {wind_choice} at SNR {args.snr} dB -> {args.out}")
        outp = mix_with_wind(args.clean, wind_choice, args.out, args.snr, target_sr=args.sr, pad_noise=args.pad_noise)
        print("Wrote:", outp)
        return

    # Directory mode
    if args.clean_dir:
        if args.out_dir is None:
            raise SystemExit("--out-dir must be provided when using --clean-dir")
        os.makedirs(args.out_dir, exist_ok=True)
        clean_files = [os.path.join(args.clean_dir, f) for f in os.listdir(args.clean_dir) if f.lower().endswith('.wav')]
        if not clean_files:
            raise SystemExit("No clean wav files found in --clean-dir")

        for clean_path, wind_path in find_wind_for_each_clean(clean_files, wind_files):
            base = os.path.basename(clean_path)
            out_path = os.path.join(args.out_dir, base)
            print(f"Mixing {base} with {os.path.basename(wind_path)} -> {out_path}")
            try:
                mix_with_wind(clean_path, wind_path, out_path, args.snr, target_sr=args.sr, pad_noise=args.pad_noise)
            except Exception as e:
                print(f"Failed for {clean_path} with {wind_path}: {e}")


if __name__ == '__main__':
    main()
