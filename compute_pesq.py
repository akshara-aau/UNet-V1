#!/usr/bin/env python3
"""
compute_pesq.py

Compute PESQ scores between clean and processed/noisy WAV files.

Features:
- Accept a single pair (clean, processed) or two directories (matching filenames) and compute PESQ for each pair.
- Resamples audio to 16000 Hz (or 8000 if requested) using librosa or scipy if available.
- Converts stereo to mono by averaging channels.
- Outputs CSV summary and prints results.

Dependencies:
- pesq (pip install pesq)
- soundfile (pip install soundfile)
- librosa (optional, used for resampling; pip install librosa) or scipy (pip install scipy)

Examples:
    # single pair
    python compute_pesq.py --clean clean.wav --processed noisy.wav --mode wb

    # directories (filename must match)
    python compute_pesq.py --clean-dir ./clean/ --processed-dir ./noisy/ --mode wb --out results_pesq.csv


"""

import os
import sys
import argparse
import csv
import logging

try:
    import soundfile as sf
except Exception:
    print("Missing dependency: soundfile. Install with `pip install soundfile`." )
    raise

try:
    from pesq import pesq
except Exception:
    print("Missing dependency: pesq. Install with `pip install pesq`. See https://pypi.org/project/pesq/ for details.")
    raise

# try optional resamplers
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

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert waveform to mono. Accepts shape (n,) or (n, channels) or (channels, n)."""
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        # prefer shape (n_samples, channels)
        if waveform.shape[0] < waveform.shape[1]:
            # likely (channels, n)
            waveform = waveform.T
        # average channels
        return waveform.mean(axis=1)
    raise ValueError("Unsupported audio shape: %s" % (waveform.shape,))


def resample_if_needed(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform to target_sr if needed. Uses librosa if available, else scipy's resample_poly.

    Raises informative error if no resampler available.
    """
    if orig_sr == target_sr:
        return waveform
    logging.info(f"Resampling from {orig_sr} Hz to {target_sr} Hz")
    if _have_librosa:
        return librosa.resample(waveform.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    if _have_scipy:
        # resample_poly works on 1D arrays
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(waveform, up, down).astype(waveform.dtype)
    raise RuntimeError("No resampler available: install librosa (`pip install librosa`) or scipy (`pip install scipy`).")


def read_audio_mono(path: str, target_sr: int = None) -> (np.ndarray, int):
    """Read audio file and return mono numpy array and sample rate. Resample to target_sr if provided."""
    data, sr = sf.read(path)
    data = to_mono(np.array(data))
    if target_sr is not None and sr != target_sr:
        data = resample_if_needed(data, sr, target_sr)
        sr = target_sr
    # ensure dtype float32
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    return data, sr


def compute_pair_pesq(clean_path: str, proc_path: str, target_sr: int = 16000, mode: str = 'wb') -> float:
    """Compute PESQ for a single pair of files.

    mode: 'wb' (wideband, 16 kHz) or 'nb' (narrowband, 8 kHz)
    target_sr: if provided, resample both to this SR before PESQ
    Returns PESQ score (float)
    """
    if mode not in ('wb', 'nb'):
        raise ValueError("mode must be 'wb' or 'nb'")
    if mode == 'nb':
        expected_sr = 8000
    else:
        expected_sr = 16000
    if target_sr is None:
        target_sr = expected_sr
    else:
        # ensure target matches mode when user didn't explicitly pass target_sr
        pass

    clean, sr_clean = read_audio_mono(clean_path, target_sr=target_sr)
    proc, sr_proc = read_audio_mono(proc_path, target_sr=target_sr)

    if sr_clean != sr_proc:
        raise RuntimeError(f"Resampled sample rates differ: {sr_clean} vs {sr_proc}")

    # trim or pad to same length - PESQ expects same length
    min_len = min(clean.shape[0], proc.shape[0])
    if min_len == 0:
        raise RuntimeError("One of the audio files is empty after reading: %s or %s" % (clean_path, proc_path))
    if clean.shape[0] != proc.shape[0]:
        logging.warning(f"Length mismatch: trimming to {min_len} samples")
        clean = clean[:min_len]
        proc = proc[:min_len]

    # pesq expects float64 in some implementations; pesq package handles float32/float64.
    try:
        score = pesq(sr_clean, clean, proc, mode)
    except Exception as e:
        # give more context
        raise RuntimeError(f"PESQ computation failed for {clean_path} vs {proc_path}: {e}")
    return float(score)


def find_matching_pairs(clean_dir: str, proc_dir: str):
    """Yield (clean_path, proc_path) for files with the same basename present in both directories."""
    clean_files = {os.path.basename(p): os.path.join(clean_dir, p) for p in os.listdir(clean_dir) if os.path.isfile(os.path.join(clean_dir, p))}
    proc_files = {os.path.basename(p): os.path.join(proc_dir, p) for p in os.listdir(proc_dir) if os.path.isfile(os.path.join(proc_dir, p))}
    common = sorted(set(clean_files.keys()) & set(proc_files.keys()))
    for name in common:
        yield clean_files[name], proc_files[name]


def main():
    parser = argparse.ArgumentParser(description="Compute PESQ between clean and processed audio files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--clean', type=str, help='Path to clean/reference wav file')
    group.add_argument('--clean-dir', type=str, help='Directory containing clean/reference wav files')
    parser.add_argument('--processed', type=str, help='Path to processed/noisy wav file (paired with --clean)')
    parser.add_argument('--processed-dir', type=str, help='Directory containing processed wav files (paired with --clean-dir)')
    parser.add_argument('--mode', type=str, default='wb', choices=['wb', 'nb'], help="PESQ mode: 'wb' (wideband, default) or 'nb' (narrowband)")
    parser.add_argument('--sr', type=int, default=None, help='Target sample rate to resample both files to before PESQ (default: 16000 for wb, 8000 for nb)')
    parser.add_argument('--out', type=str, default=None, help='CSV output path to save results')

    args = parser.parse_args()

    if args.clean and not args.processed:
        parser.error('--processed must be provided when using --clean')
    if args.clean_dir and not args.processed_dir:
        parser.error('--processed-dir must be provided when using --clean-dir')

    if args.sr is None:
        target_sr = 16000 if args.mode == 'wb' else 8000
    else:
        target_sr = args.sr

    results = []

    if args.clean:
        logging.info(f"Computing PESQ for pair: {args.clean} vs {args.processed}")
        score = compute_pair_pesq(args.clean, args.processed, target_sr=target_sr, mode=args.mode)
        results.append((os.path.basename(args.clean), score))
    else:
        # directories
        logging.info(f"Searching for matching files in {args.clean_dir} and {args.processed_dir}")
        pairs = list(find_matching_pairs(args.clean_dir, args.processed_dir))
        if not pairs:
            logging.error('No matching files found between the two directories')
            sys.exit(2)
        for clean_path, proc_path in pairs:
            logging.info(f"Processing: {os.path.basename(clean_path)}")
            try:
                score = compute_pair_pesq(clean_path, proc_path, target_sr=target_sr, mode=args.mode)
            except Exception as e:
                logging.error(f"Failed for {clean_path} vs {proc_path}: {e}")
                score = None
            results.append((os.path.basename(clean_path), score))

    # print summary
    print("\nPESQ results:")
    for name, score in results:
        print(f"{name}: {score}")

    if args.out:
        with open(args.out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'pesq'])
            for name, score in results:
                writer.writerow([name, '' if score is None else score])
        logging.info(f"Saved results to {args.out}")


if __name__ == '__main__':
    main()
