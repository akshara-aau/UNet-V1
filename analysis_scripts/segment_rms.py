import csv
import os
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


def segment_rms(audio: np.ndarray) -> float:
    """Root mean square of a 1D audio segment."""
    return float(np.sqrt(np.mean(np.square(audio))))


def load_segment(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    y, _ = librosa.load(path, sr=None, mono=True)
    return y


def list_segments(segment_dir: str) -> list[Path]:
    folder = Path(segment_dir)
    paths = list(folder.glob("segment_*.wav")) + list(folder.glob("segment_*.npy"))
    return sorted(paths, key=lambda p: int(p.stem.split("_")[1]))


def compute_rms_for_folder(segment_dir: str, output_dir: str) -> np.ndarray:
    os.makedirs(output_dir, exist_ok=True)

    paths = list_segments(segment_dir)
    if not paths:
        raise FileNotFoundError(f"No segment_*.wav or segment_*.npy in {segment_dir}")

    names: list[str] = []
    rms_values: list[float] = []

    for path in paths:
        audio = load_segment(path)
        rms = segment_rms(audio)
        names.append(path.name)
        rms_values.append(rms)

    rms_array = np.array(rms_values, dtype=np.float64)
    rms_db = 20 * np.log10(np.maximum(rms_array, 1e-10))

    stem = Path(segment_dir).name
    np.save(os.path.join(output_dir, f"{stem}_rms.npy"), rms_array)
    np.save(os.path.join(output_dir, f"{stem}_rms_db.npy"), rms_db)

    csv_path = os.path.join(output_dir, f"{stem}_rms.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "rms", "rms_db"])
        for name, rms, db in zip(names, rms_values, rms_db):
            writer.writerow([name, rms, db])

    print(f"{segment_dir}: {len(rms_values)} segments")
    print(f"  RMS  min={rms_array.min():.6f}  max={rms_array.max():.6f}  mean={rms_array.mean():.6f}")
    print(f"  saved -> {output_dir}/{stem}_rms.npy, {stem}_rms.csv")

    return rms_array, rms_db


def plot_rms_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    segment_duration_sec: float = 4.0,
) -> None:
    """Plot how RMS changes across consecutive segments."""
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    colors = {
        "segments_dynamic_clean_50": "tab:blue",
        "segments_dynamic_noisy_50": "tab:orange",
    }
    labels = {
        "segments_dynamic_clean_50": "Clean",
        "segments_dynamic_noisy_50": "Noisy",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for name, (rms, rms_db) in results.items():
        segment_idx = np.arange(len(rms))
        color = colors.get(name, None)
        label = labels.get(name, name)

        axes[0].plot(segment_idx, rms, linewidth=1.2, color=color, label=label)
        axes[1].plot(segment_idx, rms_db, linewidth=1.2, color=color, label=label)

    axes[0].set_ylabel("RMS (linear)")
    axes[0].set_title("RMS across segments")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Segment index")
    axes[1].set_ylabel("RMS (dB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path_idx = os.path.join(plot_dir, "rms_vs_segment_index.png")
    fig.savefig(path_idx, dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, (rms, rms_db) in results.items():
        n = len(rms)
        time_sec = np.arange(n) * segment_duration_sec
        color = colors.get(name, None)
        label = labels.get(name, name)

        axes2[0].plot(time_sec, rms, linewidth=1.2, color=color, label=label)
        axes2[1].plot(time_sec, rms_db, linewidth=1.2, color=color, label=label)

    axes2[0].set_ylabel("RMS (linear)")
    axes2[0].set_title("RMS across recording time")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    axes2[1].set_xlabel("Time (s)")
    axes2[1].set_ylabel("RMS (dB)")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    fig2.tight_layout()
    path_time = os.path.join(plot_dir, "rms_vs_time.png")
    fig2.savefig(path_time, dpi=150)
    plt.close(fig2)

    print(f"Plots saved -> {plot_dir}/")


if __name__ == "__main__":
    distance = "20"
    segment_dirs = [
        "segments_dynamic_clean_50",
        "segments_dynamic_noisy_50",
    ]
    output_dir = "rms-dynamic-50"
    segment_duration_sec = 4.0
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for segment_dir in segment_dirs:
        if os.path.isdir(segment_dir):
            rms, rms_db = compute_rms_for_folder(segment_dir, output_dir)
            results[segment_dir] = (rms, rms_db)
        else:
            print(f"Skipping missing folder: {segment_dir}")

    if results:
        plot_rms_curves(results, output_dir, segment_duration_sec)
