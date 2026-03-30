import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class ComplexSpeechDataset(Dataset):
    #constructor
    def __init__(self, clean_dir, noise_dir, snr_range=(-5, 15), sample_rate=16000, 
                 n_fft=512, hop_length=256, max_duration_sec=4.0):  # 4 second audio clips; n_fft=512, hop_length=256 parameters for stft; snr between -5 to 15db, fixed sampling rate of 16Khz
        super().__init__()
        # get audios files from clean and noise directory ; convert all .wav files path into list
        self.clean_files = list(Path(clean_dir).rglob("*.wav"))
        self.noise_files = list(Path(noise_dir).rglob("*.wav"))
        
        self.snr_range = snr_range
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_sec) 
        # returning complex stft; [channel0=real, channel1=imaginary, frequency bins, time frames]; model to learn phase and magnitude
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
        )
    # tell pyTorch how many samples are in the dataset; it will decide the batch
    def __len__(self):
        return len(self.clean_files)
    # return the length standardise tensor, either pad or truncate
    def _pad_or_truncate(self, waveform):
        if waveform.shape[1] > self.max_length:
            start = random.randint(0, waveform.shape[1] - self.max_length)
            waveform = waveform[:, start:start + self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Independent waveform peak-norm before mixing
        max_amp = torch.max(torch.abs(waveform)) + 1e-8
        waveform = waveform / max_amp
            
        return self._pad_or_truncate(waveform)
    # apply reverbration to clean speech ; faking room echo ; egenralize to real spaces
    def _apply_reverb(self, waveform):
        # 50% chance to apply reverb
        if random.random() < 0.5:
            # Generate random synthetic RT60 Room Impulse Response (0.1s to 0.6s)
            rt60 = random.uniform(0.1, 0.6)
            rir_length = int(self.sample_rate * rt60)
            
            # White noise decaying exponentially to -60dB
            decay = 6.908 / rir_length
            envelope = torch.exp(-decay * torch.arange(rir_length, dtype=torch.float32))
            rir = torch.randn(1, rir_length) * envelope
            
            # Normalize RIR so it doesn't change overall volume drastically
            rir = rir / torch.norm(rir, p=2)
            
            # Apply FFT Convolve
            reverbed = torchaudio.functional.fftconvolve(waveform, rir)
            
            # Trim the convolution tail
            waveform = reverbed[:, :waveform.shape[1]]
        return waveform

    def __getitem__(self, idx):
        clean_waveform_dry = self._load_audio(self.clean_files[idx])
        noise_waveform = self._load_audio(random.choice(self.noise_files))
        
        # Apply synthetic reverb to the clean speech before mixing
        reverbed_clean = self._apply_reverb(clean_waveform_dry)
        
        snr_db = random.uniform(*self.snr_range)
        eps = 1e-8
        
        clean_energy = torch.mean(reverbed_clean ** 2)
        noise_energy = torch.mean(noise_waveform ** 2)
        
        target_noise_energy = clean_energy / (10 ** (snr_db / 10) + eps)
        noise_scalar = torch.sqrt(target_noise_energy / (noise_energy + eps))
        scaled_noise = noise_waveform * noise_scalar
        
        mixture = reverbed_clean + scaled_noise
        
        # Unconditional peak-normalization for the mixture to guarantee consistent ML input scale
        max_val = torch.max(torch.abs(mixture)) + eps
        mixture = mixture / max_val
        clean_waveform_dry = clean_waveform_dry / max_val

        # Get Complex Tensors [Channels, Freq, Time]
        mix_stft = self.stft(mixture)
        clean_stft = self.stft(clean_waveform_dry)
        
        # We must separate the complex output into Real and Imaginary for the PyTorch Conv layers!
        # Tanh layers in the model want data between -1 and 1. 
        # For simplicity in this demo class, we will normalize the spectrograms.
        normalize_factor = torch.max(torch.abs(mix_stft)) + eps
        
        mix_real = mix_stft.real / normalize_factor
        mix_imag = mix_stft.imag / normalize_factor
        
        clean_real = clean_stft.real / normalize_factor
        clean_imag = clean_stft.imag / normalize_factor
        
        # Add channel dimensions manually so it is [1, Freq, Time]
        return mix_real, mix_imag, clean_real, clean_imag

def get_dataloaders(clean_dir, noise_dir, batch_size=16, snr_range=(-5, 15)):
    dataset = ComplexSpeechDataset(clean_dir, noise_dir, snr_range)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader
