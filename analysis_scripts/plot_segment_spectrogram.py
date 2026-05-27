"""
Plot spectrogram for an audio file.

Usage:
    python plot_segment_spectrogram.py path/to/audio.wav
    python plot_segment_spectrogram.py recording.flac -o spectrogram.png
    python plot_segment_spectrogram.py clip.wav --start 4 --end 9 -o spec_4_9s.png
"""

import argparse
import os
from pathlib import Path

import librosa
import librosa.display
# import matplotlib_parula  # registers 'parula' colormap
import matplotlib.pyplot as plt
import numpy as np

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}


def resolve_output_path(path: Path) -> Path:
    """Ensure output path has an image extension."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return path.with_suffix(".png")
    return path


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio; returns (mono waveform, sample_rate)."""
    if path.suffix == ".npy":
        audio = np.load(path).astype(np.float64)
        if sr is None:
            raise ValueError("Pass --sr when input is a .npy file")
        return audio, sr
    if path.suffix.lower() not in AUDIO_EXTENSIONS and path.suffix != ".npy":
        raise ValueError(
            f"Unsupported format {path.suffix}. "
            f"Use one of: {', '.join(sorted(AUDIO_EXTENSIONS))} or .npy"
        )
    audio, loaded_sr = librosa.load(path, sr=sr, mono=True)
    return audio, loaded_sr


def slice_by_time(
    audio: np.ndarray,
    sr: int,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[np.ndarray, float, float]:
    """Extract [start_sec, end_sec) from audio; returns (slice, start, end)."""
    duration = len(audio) / sr
    start = 0.0 if start_sec is None else start_sec
    end = duration if end_sec is None else end_sec

    if start < 0 or end <= start:
        raise ValueError(f"Invalid time range: start={start}, end={end}")
    if start >= duration:
        raise ValueError(
            f"start={start}s is beyond audio duration ({duration:.2f}s)"
        )

    end = min(end, duration)
    start_idx = int(start * sr)
    end_idx = int(end * sr)
    return audio[start_idx:end_idx], start, end


def compute_spectrogram_db(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
) -> tuple[np.ndarray, int, int]:
    # Periodic Hann window — matches MATLAB hann(winLen, 'periodic')
    window = np.hanning(win_length + 1)[:-1]
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=window))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db, n_fft, hop_length


def figure_size_for_spec(n_frames: int, duration_sec: float) -> tuple[float, float]:
    """Size figure so time bins are not over-stretched (reduces blur)."""
    width = max(8.0, min(24.0, duration_sec * 2.0, n_frames / 80.0))
    return width, 5.0


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str,
    output_path: str | None = None,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    time_offset_sec: float = 0.0,
    vmin: float | None = -60.0,
    vmax: float | None = 0.0,
    dpi: int = 300,
    show: bool = True,
) -> None:
    """Plot STFT spectrogram."""
    S_db, n_fft, hop_length = compute_spectrogram_db(audio, sr, n_fft, hop_length, win_length)
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length) + time_offset_sec
    duration_sec = len(audio) / sr
    fig_w, fig_h = figure_size_for_spec(S_db.shape[1], duration_sec)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    specshow_kw: dict = {"cmap": "jet"}
    if vmin is not None:
        specshow_kw["vmin"] = vmin
    if vmax is not None:
        specshow_kw["vmax"] = vmax

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        x_coords=times,
        ax=ax,
        **specshow_kw,
    )
    if hasattr(img, "set_interpolation"):
        img.set_interpolation("nearest")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        print(f"Saved -> {output_path} ({dpi} dpi)")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot spectrogram of an audio file.")
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to input audio (wav, flac, mp3, etc.) or .npy waveform",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Target sample rate (default: native rate for audio files; required for .npy)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="File name to save the spectrogram (e.g. spectrogram.png or plots/out)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open plot window (with -o: also display after saving)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds (e.g. 4)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (e.g. 9)",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=512,
        help="FFT size (larger = sharper frequency detail, default: 512)",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="STFT hop (smaller = sharper time detail, default: 256)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-60.0,
        help="Min dB for color scale (default: -60)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.0,
        help="Max dB for color scale (default: 0)",
    )
    parser.add_argument(
        "--no-vmin",
        action="store_true",
        help="Auto-scale dB range (librosa default)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image resolution (default: 300)",
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    audio, sr = load_audio(args.audio, sr=args.sr)
    audio, t_start, t_end = slice_by_time(audio, sr, args.start, args.end)

    if len(audio) == 0:
        raise ValueError("Selected time range is empty")

    output_path = None
    if args.output is not None:
        output_path = str(resolve_output_path(args.output))

    show = args.show or output_path is None

    vmin = None if args.no_vmin else args.vmin
    vmax = None if args.no_vmin else args.vmax

    plot_spectrogram(
        audio,
        sr,
        title=args.audio.name,
        output_path=output_path,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.n_fft,
        time_offset_sec=t_start,
        vmin=vmin,
        vmax=vmax,
        dpi=args.dpi,
        show=show,
    )


if __name__ == "__main__":
    main()