#!/usr/bin/env python3
"""
compare_phase.py

Compare phase between two audio files.

This script computes STFTs for two wav files (reference and test), extracts the phase
spectrograms and computes several comparison metrics:

- Phase difference spectrogram (wrapped to [-pi, pi])
- Per-frequency Phase Locking Value (PLV): |<exp(i * delta_phase)>_time|
- Mean circular phase difference (global)
- Per-frequency circular mean phase difference and resultant length
- Optional plots: heatmap of phase difference and PLV curve

Usage:
    python compare_phase.py ref.wav test.wav --plot out.png --n_fft 1024 --hop_length 256

Dependencies: soundfile, numpy, matplotlib (optional), librosa or scipy for resampling/STFT.
"""

import argparse
import os
import sys
from typing import Tuple, Optional

import numpy as np
import soundfile as sf

# try librosa for STFT/resampling; fallback to scipy
_have_librosa = False
_have_scipy = False
try:
    import librosa
    _have_librosa = True
except Exception:
    try:
        from scipy.signal import stft, resample_poly
        _have_scipy = True
    except Exception:
        pass

# optional plotting
_try_plot = False
try:
    import matplotlib.pyplot as plt
    _try_plot = True
except Exception:
    _try_plot = False


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        # assume (n_samples, channels) as returned by soundfile
        if y.shape[0] < y.shape[1]:
            y = y.T
        return y.mean(axis=1)
    raise ValueError(f"Unsupported audio shape: {y.shape}")


def resample_if_needed(y: np.ndarray, orig_sr: int, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
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
    raise RuntimeError("No resampler available: install librosa or scipy")


def read_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32")
    data = to_mono(np.asarray(data))
    data, sr = resample_if_needed(data, sr, target_sr)
    return data.astype(np.float32), sr


def compute_stft(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Return complex spectrogram S (shape freq bins x frames) and frequency vector"""
    if _have_librosa:
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
        freqs = np.linspace(0, sr / 2, num=S.shape[0])
        return S, freqs
    if _have_scipy:
        # scipy.signal.stft returns f, t, Zxx where Zxx is shape (freqs, frames)
        f, t, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=(n_fft - hop_length), boundary=None)
        return Zxx, f
    raise RuntimeError("No STFT implementation available: install librosa or scipy")


def circular_mean_complex(angles: np.ndarray, axis=None) -> complex:
    """Compute circular mean as complex mean of exp(i*angle). Returns complex mean (angle via np.angle)."""
    c = np.mean(np.exp(1j * angles), axis=axis)
    return c


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi]"""
    return (phase + np.pi) % (2 * np.pi) - np.pi


def compare_phase(ref_path: str, test_path: str, n_fft: int = 1024, hop_length: int = 256, target_sr: Optional[int] = None, plot_path: Optional[str] = None):
    # Read without forcing resampling first so we can decide a sensible common SR
    ref, sr_ref = read_audio(ref_path, target_sr=None)
    test, sr_test = read_audio(test_path, target_sr=None)

    # If user explicitly provided target_sr, resample both to that. Otherwise, if the files
    # have different sample rates, pick a common rate (prefer the higher SR) and resample both.
    if target_sr is not None:
        ref, _ = resample_if_needed(ref, sr_ref, target_sr)
        test, _ = resample_if_needed(test, sr_test, target_sr)
        sr = target_sr
    else:
        if sr_ref != sr_test:
            # choose common SR: prefer the higher sampling rate to avoid losing information
            common_sr = max(sr_ref, sr_test)
            print(f"Warning: input sample rates differ ({sr_ref} vs {sr_test}). Resampling both to {common_sr} Hz for comparison.")
            ref, _ = resample_if_needed(ref, sr_ref, common_sr)
            test, _ = resample_if_needed(test, sr_test, common_sr)
            sr = common_sr
        else:
            sr = sr_ref

    # trim to same length
    min_len = min(len(ref), len(test))
    if min_len == 0:
        raise RuntimeError("One of the input files is empty")
    if len(ref) != len(test):
        ref = ref[:min_len]
        test = test[:min_len]

    S_ref, freqs = compute_stft(ref, sr, n_fft=n_fft, hop_length=hop_length)
    S_test, _ = compute_stft(test, sr, n_fft=n_fft, hop_length=hop_length)

    # shapes should match
    if S_ref.shape != S_test.shape:
        # attempt to align frames by min shape
        min_bins = min(S_ref.shape[0], S_test.shape[0])
        min_frames = min(S_ref.shape[1], S_test.shape[1])
        S_ref = S_ref[:min_bins, :min_frames]
        S_test = S_test[:min_bins, :min_frames]
        freqs = freqs[:min_bins]

    phase_ref = np.angle(S_ref)
    phase_test = np.angle(S_test)

    # phase difference wrapped
    delta = wrap_phase(phase_test - phase_ref)

    # per-frequency PLV: magnitude of mean phase-vector across time
    plv = np.abs(circular_mean_complex(delta, axis=1))  # shape: (freqs,)

    # per-frequency mean angle and resultant length
    mean_complex_per_freq = circular_mean_complex(delta, axis=1)
    mean_angle_per_freq = np.angle(mean_complex_per_freq)
    resultant_length_per_freq = np.abs(mean_complex_per_freq)

    # global circular mean
    global_complex = circular_mean_complex(delta)
    global_mean_angle = np.angle(global_complex)
    global_resultant_length = np.abs(global_complex)

    # per-frequency unwrapped correlation (pearson) between ref and test phases
    # unwrap across time for each frequency then compute correlation
    unwrapped_ref = np.unwrap(phase_ref, axis=1)
    unwrapped_test = np.unwrap(phase_test, axis=1)
    corr_per_freq = np.array([
        np.corrcoef(unwrapped_ref[i, :], unwrapped_test[i, :])[0, 1] if unwrapped_ref.shape[1] > 1 else 0.0
        for i in range(unwrapped_ref.shape[0])
    ])

    # summary
    summary = {
        "global_mean_phase_diff_rad": float(global_mean_angle),
        "global_resultant_length": float(global_resultant_length),
        "n_freq_bins": int(freqs.shape[0]),
        "n_frames": int(phase_ref.shape[1]),
        "sr": int(sr),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
    }

    # Optionally plot: show phase spectrograms for ref, test, and their difference
    if plot_path:
        if not _try_plot:
            print("matplotlib not available; skipping plot")
        else:
            n_bins, n_frames = delta.shape
            # time axis in seconds based on hop_length and sr
            times = np.arange(n_frames) * (hop_length / float(sr))

            def _save_phase_image(phase_arr, out_path, title):
                fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
                im = ax.imshow(wrap_phase(phase_arr), aspect='auto', origin='lower', cmap='twilight',
                               vmin=-np.pi, vmax=np.pi,
                               extent=[times[0] if len(times) else 0, times[-1] if len(times) else 0, freqs[0], freqs[-1]])
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Time (s)')
                ax.set_title(title)
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                cbar.set_label('radians')
                plt.savefig(out_path)
                plt.close(fig)

            # Save separate phase images for reference (noisy) and test (noise-suppressed)
            base, ext = os.path.splitext(plot_path)
            ref_out = f"{base}_ref{ext}"
            test_out = f"{base}_test{ext}"
            _save_phase_image(phase_ref, ref_out, f"Phase (reference/noisy): {os.path.basename(ref_path)}")
            _save_phase_image(phase_test, test_out, f"Phase (test/suppressed): {os.path.basename(test_path)}")
            print(f"Saved reference phase to {ref_out}")
            print(f"Saved test phase to {test_out}")

            # Combined 3-panel figure for convenience
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
            im0 = axes[0].imshow(wrap_phase(phase_ref), aspect='auto', origin='lower', cmap='twilight',
                                  vmin=-np.pi, vmax=np.pi,
                                  extent=[times[0] if len(times) else 0, times[-1] if len(times) else 0, freqs[0], freqs[-1]])
            axes[0].set_ylabel('Frequency (Hz)')
            axes[0].set_title('Phase: reference (noisy)')
            fig.colorbar(im0, ax=axes[0], orientation='vertical', pad=0.01)

            im1 = axes[1].imshow(wrap_phase(phase_test), aspect='auto', origin='lower', cmap='twilight',
                                  vmin=-np.pi, vmax=np.pi,
                                  extent=[times[0] if len(times) else 0, times[-1] if len(times) else 0, freqs[0], freqs[-1]])
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_title('Phase: test (noise-suppressed)')
            fig.colorbar(im1, ax=axes[1], orientation='vertical', pad=0.01)

            im2 = axes[2].imshow(delta, aspect='auto', origin='lower', cmap='RdBu_r',
                                  vmin=-np.pi, vmax=np.pi,
                                  extent=[times[0] if len(times) else 0, times[-1] if len(times) else 0, freqs[0], freqs[-1]])
            axes[2].set_ylabel('Frequency (Hz)')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_title('Phase difference (test - ref)')
            fig.colorbar(im2, ax=axes[2], orientation='vertical', pad=0.01)

            fig.suptitle(f'Phase comparison: {os.path.basename(ref_path)} vs {os.path.basename(test_path)}')
            plt.savefig(plot_path)
            plt.close(fig)
            print(f'Saved combined plot to {plot_path}')

    # Per-frequency stats exported as arrays
    per_freq = {
        "freqs": freqs,
        "plv": plv,
        "mean_angle_per_freq": mean_angle_per_freq,
        "resultant_length_per_freq": resultant_length_per_freq,
        "corr_per_freq": corr_per_freq,
    }

    return summary, per_freq


def main():
    parser = argparse.ArgumentParser(description="Compare phase between two audio files")
    parser.add_argument("ref", help="Reference (clean) wav file")
    parser.add_argument("test", help="Test wav file")
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--sr", type=int, default=None, help="Target sample rate (optional)")
    parser.add_argument("--plot", type=str, default=None, help="Path to save plot PNG")
    parser.add_argument("--out-csv", type=str, default=None, help="Path to save per-frequency CSV")

    args = parser.parse_args()

    summary, per_freq = compare_phase(args.ref, args.test, n_fft=args.n_fft, hop_length=args.hop_length, target_sr=args.sr, plot_path=args.plot)

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["freq", "plv", "mean_angle_rad", "resultant_length", "corr"])
            for i in range(len(per_freq["freqs"])):
                writer.writerow([
                    float(per_freq["freqs"][i]),
                    float(per_freq["plv"][i]),
                    float(per_freq["mean_angle_per_freq"][i]),
                    float(per_freq["resultant_length_per_freq"][i]),
                    float(per_freq["corr_per_freq"][i] if np.isfinite(per_freq["corr_per_freq"][i]) else 0.0),
                ])
        print(f"Wrote per-frequency metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
