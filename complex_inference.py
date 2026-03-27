import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchaudio
import soundfile as sf
import numpy as np

from complex_model import DeepComplexUNet


def enhance_audio(noisy_wav_path, model_path,
                  output_path="enhanced_output.wav",
                  model_size='dcu16',
                  n_fft=512, hop_length=256,
                  target_sr=16000):
    """
    Enhance a single noisy audio file using a trained DCUNet model.

    Args:
        noisy_wav_path : path to the noisy input .wav file
        model_path     : path to the saved model checkpoint (.pth)
        output_path    : where to save the enhanced .wav
        model_size     : must match the size used during training
        n_fft          : must match training STFT settings
        hop_length     : must match training STFT settings
        target_sr      : sample rate (model was trained at 16000)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load model ----
    if not os.path.exists(model_path):
        print(f"ERROR: No model file found at '{model_path}'")
        return

    model = DeepComplexUNet(n_channels=1, model_size=model_size)
    ckpt  = torch.load(model_path, map_location=device, weights_only=True)

    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"Model loaded: {model_path}  ({model_size})")

    # ---- Load audio ----
    if not os.path.exists(noisy_wav_path):
        print(f"ERROR: File not found: '{noisy_wav_path}'")
        return

    audio_data, sr = sf.read(noisy_wav_path)

    # Stereo -> mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    waveform = torch.FloatTensor(audio_data).unsqueeze(0)  # [1, T]

    # Resample if necessary
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        print(f"Resampled {sr}Hz -> {target_sr}Hz")

    # Peak normalise — same operation as in training data pipeline
    peak = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / peak

    # ---- STFT ----
    stft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
    ).to(device)
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft, hop_length=hop_length, normalized=True
    ).to(device)

    complex_spec = stft(waveform.to(device))          # [1, F, T] complex

    norm     = torch.max(torch.abs(complex_spec)) + 1e-8
    mix_real = (complex_spec.real / norm).unsqueeze(0).to(device)  # [1, 1, F, T]
    mix_imag = (complex_spec.imag / norm).unsqueeze(0).to(device)

    # ---- Inference ----
    with torch.no_grad():
        mask_r, mask_i = model(mix_real, mix_imag)

    # Apply Complex Ratio Mask — this is the step that was missing before
    #   est = mask * noisy_spec  (complex multiplication)
    #   (M_r + j*M_i)(Y_r + j*Y_i) = (M_r*Y_r - M_i*Y_i) + j(M_r*Y_i + M_i*Y_r)
    est_real = mask_r * mix_real - mask_i * mix_imag
    est_imag = mask_r * mix_imag + mask_i * mix_real

    # Denormalise
    est_real = est_real.squeeze(0) * norm
    est_imag = est_imag.squeeze(0) * norm

    # ---- ISTFT ----
    enhanced_complex  = torch.complex(est_real, est_imag)      # [1, F, T]
    enhanced_waveform = istft(enhanced_complex).squeeze().cpu().numpy()

    # Final peak normalise for output
    max_out = np.max(np.abs(enhanced_waveform)) + 1e-8
    enhanced_waveform = enhanced_waveform / max_out

    # ---- Save ----
    sf.write(output_path, enhanced_waveform, target_sr)
    print(f"Enhanced audio saved to: {output_path}")


def batch_enhance(input_dir, model_path, output_dir,
                  model_size='dcu16', **kwargs):
    """
    Enhance all .wav files in input_dir and save to output_dir.
    """
    from pathlib import Path
    os.makedirs(output_dir, exist_ok=True)

    wav_files = list(Path(input_dir).rglob("*.wav"))
    print(f"Enhancing {len(wav_files)} files...")

    for wav_path in wav_files:
        out_path = os.path.join(output_dir, wav_path.name)
        enhance_audio(
            noisy_wav_path=str(wav_path),
            model_path=model_path,
            output_path=out_path,
            model_size=model_size,
            **kwargs
        )

    print("Done.")


if __name__ == "__main__":
    # ---- Single file enhancement ----
    enhance_audio(
        noisy_wav_path="test_audio/noisy_sample.wav",
        model_path="best_model.pth",
        output_path="test_audio/enhanced_sample.wav",
        model_size='dcu16',   # must match what you trained
    )

    # ---- Batch enhancement (uncomment to use) ----
    # batch_enhance(
    #     input_dir="test_audio/noisy/",
    #     model_path="best_model.pth",
    #     output_dir="test_audio/enhanced/",
    #     model_size='dcu16',
    # )