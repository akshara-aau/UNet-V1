import os
import torch
import torchaudio
import random
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class ComplexSpeechDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, rir_dir=None, snr_range=(-5, 15), sample_rate=16000, 
                 n_fft=512, hop_length=256, max_duration_sec=4.0):
        super().__init__()
        self.clean_files = list(Path(clean_dir).rglob("*.wav"))
        self.noise_files = list(Path(noise_dir).rglob("*.wav"))
        self.rir_files = []
        if rir_dir and os.path.exists(rir_dir):
            self.rir_files = list(Path(rir_dir).rglob("*.wav"))
        
        self.snr_range = snr_range
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_sec) 
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
        )

    def __len__(self):
        return len(self.clean_files)

    def _pad_or_truncate(self, waveform):
        if waveform.shape[1] > self.max_length:
            start = random.randint(0, waveform.shape[1] - self.max_length)
            waveform = waveform[:, start:start + self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def _load_audio(self, path):
        for _ in range(3):  # Try up to 3 times with different random files if we hit a silent one
            try:
                data, sr = sf.read(path, dtype='float32')
                if data.ndim == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(data.T)
                    
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Check for "Dead/Silent" audio to prevent NaN loss
                max_amp = torch.max(torch.abs(waveform))
                if max_amp < 1e-6:
                    # If file is silent, pick a random new one and try again
                    path = random.choice(self.clean_files)
                    continue

                waveform = waveform / (max_amp + 1e-8)
                return self._pad_or_truncate(waveform)
            except:
                path = random.choice(self.clean_files)
        
        return torch.zeros((1, self.max_length))

    def _apply_reverb(self, waveform):
        if self.rir_files and random.random() < 0.5:
            try:
                rir_path = random.choice(self.rir_files)
                data, sr = sf.read(rir_path, dtype='float32')
                rir_waveform = torch.from_numpy(data.T) if data.ndim > 1 else torch.from_numpy(data).unsqueeze(0)
                if sr != self.sample_rate:
                    rir_waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(rir_waveform)
                rir_waveform = rir_waveform[:1, :]
                rir_waveform = rir_waveform / (torch.max(torch.abs(rir_waveform)) + 1e-8)
                reverbed = torchaudio.functional.fftconvolve(waveform, rir_waveform)
                waveform = reverbed[:, :waveform.shape[1]]
            except:
                pass
        return waveform

    def __getitem__(self, idx):
        clean_waveform_dry = self._load_audio(self.clean_files[idx])
        # Also ensure noise isn't silent
        noise_waveform = self._load_audio(random.choice(self.noise_files))
        
        reverbed_clean = self._apply_reverb(clean_waveform_dry)
        snr_db = random.uniform(*self.snr_range)
        eps = 1e-8
        
        clean_energy = torch.mean(reverbed_clean ** 2)
        noise_energy = torch.mean(noise_waveform ** 2)
        
        target_noise_energy = clean_energy / (10 ** (snr_db / 10) + eps)
        noise_scalar = torch.sqrt(target_noise_energy / (noise_energy + eps))
        scaled_noise = noise_waveform * noise_scalar
        
        mixture = reverbed_clean + scaled_noise
        
        # Unconditional normalization
        max_val = torch.max(torch.abs(mixture)) + eps
        mixture = mixture / max_val
        clean_waveform_dry = clean_waveform_dry / max_val
        
        mix_stft = self.stft(mixture)
        clean_stft = self.stft(clean_waveform_dry)
        
        normalize_factor = torch.max(torch.abs(mix_stft)) + eps
        mix_real = mix_stft.real / normalize_factor
        mix_imag = mix_stft.imag / normalize_factor
        clean_real = clean_stft.real / normalize_factor
        clean_imag = clean_stft.imag / normalize_factor
        
        return mix_real, mix_imag, clean_real, clean_imag

def get_dataloaders(clean_dir, noise_dir, rir_dir=None, batch_size=16, snr_range=(-5, 15)):
    dataset = ComplexSpeechDataset(clean_dir, noise_dir, rir_dir, snr_range)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader
