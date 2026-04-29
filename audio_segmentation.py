
import os
from typing import List

import numpy as np
import soundfile as sf


def _to_mono(y: np.ndarray) -> np.ndarray:
    """Convert audio to mono. Accept shapes (n,) or (n, c) or (c, n)."""
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        # assume (n_samples, channels) is standard for soundfile
        if y.shape[0] < y.shape[1]:
            y = y.T
        return y.mean(axis=1)
    raise ValueError(f"Unsupported audio shape: {y.shape}")


def segment_audio(
    audio_path: str,
    output_dir: str,
    segment_duration: int = 60,
    overlap: float = 0.0,
    pad_last: bool = True,
    skip_silence: bool = False,
    silence_threshold: float = 0.01,
) -> List[str]:
    """Split an audio file into fixed-length segments and save them as WAV files.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory where segments will be written. Created if missing.
        segment_duration: Segment length in seconds (default 30).
        overlap: Fractional overlap between 0 and <1 (default 0.0 no overlap).
        pad_last: If True, pad the last segment with zeros to full length; else keep shorter last segment.
        skip_silence: If True, skip segments whose max absolute amplitude < silence_threshold.
        silence_threshold: Threshold for silence skipping (float in same scale as waveform).

    Returns:
        List of file paths written (in order).
    """
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1)")

    audio_data, sr = sf.read(audio_path ,stop= 3600 * 16000) 
    y = np.asarray(audio_data)
    y = _to_mono(y)

    segment_length = int(segment_duration * sr)
    if segment_length <= 0:
        raise ValueError("segment_duration too small for sample rate")

    hop_length = int(segment_length * (1 - overlap))
    if hop_length <= 0:
        raise ValueError("overlap too large, resulting in non-positive hop length")

    os.makedirs(output_dir, exist_ok=True)
    written_paths: List[str] = []

    n_samples = y.shape[0]
    starts = list(range(0, max(1, n_samples - segment_length + 1), hop_length))

    count = 0
    for start in starts:
        end = start + segment_length
        seg = y[start:end]
        if seg.shape[0] < segment_length:
            if pad_last:
                pad_width = segment_length - seg.shape[0]
                seg = np.pad(seg, (0, pad_width), mode="constant")
            # else leave shorter

        if skip_silence and np.max(np.abs(seg)) < silence_threshold:
            continue

        out_path = os.path.join(output_dir, f"segment_{count:05d}.wav")
        # write as float32
        sf.write(out_path, seg.astype(np.float32), sr)
        written_paths.append(out_path)
        count += 1

    # handle case where audio is shorter than one segment
    if n_samples > 0 and len(written_paths) == 0:
        seg = y
        if pad_last and seg.shape[0] < segment_length:
            seg = np.pad(seg, (0, segment_length - seg.shape[0]), mode="constant")
        out_path = os.path.join(output_dir, f"segment_{0:05d}.wav")
        sf.write(out_path, seg.astype(np.float32), sr)
        written_paths.append(out_path)

    print(f"Total segments created: {len(written_paths)} (saved to {output_dir})")
    return written_paths

