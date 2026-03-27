import os
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pyroomacoustics as pra
    PRA_AVAILABLE = True
except ImportError:
    PRA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset: MS-SNSD style (synthesises pairs on the fly)
# Use this if you are training on MS-SNSD or similar raw clean+noise dirs
# ---------------------------------------------------------------------------

class ComplexSpeechDataset(Dataset):
    """
    On-the-fly noisy speech synthesis with optional RIR augmentation.

    Directory layout expected:
        clean_dir/  *.wav   (clean speech files)
        noise_dir/  *.wav   (noise files, any subdirs)
        rir_dir/    *.wav   (optional room impulse responses)

    Args:
        clean_dir       : path to clean speech .wav files
        noise_dir       : path to noise .wav files
        rir_dir         : path to room impulse response .wav files (optional)
        snr_range       : (min_dB, max_dB) for mixing  — use (0, 40) for DNS-style
        sample_rate     : target sample rate (16000)
        n_fft           : STFT FFT size (512)
        hop_length      : STFT hop length (256)
        max_duration_sec: clip length in seconds (4.0)
        rir_prob        : probability of applying RIR per sample (0.7)
    """

    def __init__(self, clean_dir, noise_dir, rir_dir=None,
                 snr_range=(0, 40), sample_rate=16000,
                 n_fft=512, hop_length=256, max_duration_sec=4.0,
                 rir_prob=0.7):
        super().__init__()
        self.clean_files = list(Path(clean_dir).rglob("*.wav"))
        self.noise_files = list(Path(noise_dir).rglob("*.wav"))
        self.rir_files   = list(Path(rir_dir).rglob("*.wav")) if rir_dir else []

        assert len(self.clean_files) > 0, f"No .wav files found in {clean_dir}"
        assert len(self.noise_files) > 0, f"No .wav files found in {noise_dir}"

        self.snr_range       = snr_range
        self.sample_rate     = sample_rate
        self.max_length      = int(sample_rate * max_duration_sec)
        self.rir_prob        = rir_prob

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
        )

        print(f"Dataset: {len(self.clean_files)} clean | "
              f"{len(self.noise_files)} noise | "
              f"{len(self.rir_files)} RIRs | "
              f"SNR range {snr_range} dB")

    def __len__(self):
        return len(self.clean_files)

    # ------------------------------------------------------------------
    # Audio loading helpers
    # ------------------------------------------------------------------

    def _pad_or_truncate(self, waveform):
        """Randomly crop or zero-pad to exactly self.max_length samples."""
        T = waveform.shape[1]
        if T > self.max_length:
            start = random.randint(0, T - self.max_length)
            waveform = waveform[:, start:start + self.max_length]
        elif T < self.max_length:
            pad = self.max_length - T
            waveform = F_pad(waveform, (0, pad))
        return waveform

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:                     # stereo -> mono
            waveform = waveform.mean(dim=0, keepdim=True)
        return self._pad_or_truncate(waveform)

    # ------------------------------------------------------------------
    # RIR augmentation
    # ------------------------------------------------------------------

    def _apply_rir(self, waveform):
        """
        Convolve waveform with a random room impulse response.
        Falls back to pyroomacoustics simulation if no RIR files are available.
        Returns the reverberant waveform at the same length.
        """
        if random.random() > self.rir_prob:
            return waveform

        rir_np = None

        # Option A: use a real RIR file
        if self.rir_files and SCIPY_AVAILABLE:
            try:
                rir_wav, sr = torchaudio.load(random.choice(self.rir_files))
                if sr != self.sample_rate:
                    rir_wav = torchaudio.transforms.Resample(sr, self.sample_rate)(rir_wav)
                rir_np = rir_wav[0].numpy()
            except Exception:
                rir_np = None

        # Option B: simulate a room with pyroomacoustics
        if rir_np is None and PRA_AVAILABLE:
            try:
                rir_np = self._simulate_rir()
            except Exception:
                rir_np = None

        if rir_np is None or not SCIPY_AVAILABLE:
            return waveform   # skip augmentation gracefully

        wav_np = waveform[0].numpy()
        reverberant = scipy.signal.fftconvolve(wav_np, rir_np)[:len(wav_np)]
        # Prevent NaN/Inf from bad RIR files
        if not np.isfinite(reverberant).all():
            return waveform
        return torch.FloatTensor(reverberant).unsqueeze(0)

    def _simulate_rir(self):
        """Generate a random shoebox room RIR using pyroomacoustics."""
        Lx = random.uniform(3, 10)
        Ly = random.uniform(3, 8)
        Lz = random.uniform(2.5, 4.0)
        absorption = random.uniform(0.1, 0.8)

        room = pra.ShoeBox(
            [Lx, Ly, Lz], fs=self.sample_rate,
            materials=pra.Material(absorption),
            max_order=17
        )
        src = [random.uniform(0.5, Lx - 0.5),
               random.uniform(0.5, Ly - 0.5), 1.5]
        mic = [random.uniform(0.5, Lx - 0.5),
               random.uniform(0.5, Ly - 0.5), 1.5]
        room.add_source(src)
        room.add_microphone(np.array(mic).reshape(3, 1))
        room.compute_rir()
        return room.rir[0][0]

    # ------------------------------------------------------------------
    # Core item generation
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        clean = self._load_audio(self.clean_files[idx])
        noise = self._load_audio(random.choice(self.noise_files))

        # Apply RIR to clean speech (model must denoise reverberant speech)
        clean_reverb = self._apply_rir(clean)

        # Mix at random SNR
        snr_db = random.uniform(*self.snr_range)
        eps    = 1e-8

        clean_energy = clean_reverb.pow(2).mean()
        noise_energy = noise.pow(2).mean()

        # Skip near-silent clips to avoid NaN scaling
        if clean_energy < 1e-6 or noise_energy < 1e-6:
            return self.__getitem__((idx + 1) % len(self))

        target_noise_energy = clean_energy / (10 ** (snr_db / 10))
        noise_scalar        = torch.sqrt(target_noise_energy / (noise_energy + eps))
        scaled_noise        = noise * noise_scalar

        mixture = clean_reverb + scaled_noise

        # Peak normalise the mixture — same operation as inference
        # This is what allows the model to generalise to any volume level
        peak = torch.max(torch.abs(mixture)) + eps
        mixture      = mixture / peak
        clean_reverb = clean_reverb / peak   # scale clean by SAME factor

        # Compute complex STFTs
        mix_stft   = self.stft(mixture)
        clean_stft = self.stft(clean_reverb)

        # Normalise spectrograms by mixture maximum
        # Never use clean max — that leaks target info into the input scale
        norm = torch.max(torch.abs(mix_stft)) + eps

        mix_real   = mix_stft.real   / norm
        mix_imag   = mix_stft.imag   / norm
        clean_real = clean_stft.real / norm
        clean_imag = clean_stft.imag / norm

        return mix_real, mix_imag, clean_real, clean_imag


# ---------------------------------------------------------------------------
# Dataset: DNS pre-mixed pairs (use this when training on Microsoft DNS data)
# ---------------------------------------------------------------------------

class DNSSpeechDataset(Dataset):
    """
    Reads pre-mixed (noisy, clean) pairs generated by Microsoft's
    noisyspeech_synthesizer.py from the DNS-Challenge repository.

    Directory layout expected:
        noisy_dir/  noisy_fileidXXXXX_snrYY_tgt...wav
        clean_dir/  clean_fileidXXXXX_snrYY_tgt...wav
        (files paired by matching filename after 'fileid')

    Args:
        noisy_dir : path to noisy .wav files
        clean_dir : path to clean .wav files
    """

    def __init__(self, noisy_dir, clean_dir,
                 sample_rate=16000, n_fft=512, hop_length=256,
                 max_duration_sec=4.0):
        super().__init__()

        noisy_files = {f.stem: f for f in Path(noisy_dir).rglob("*.wav")}
        clean_files = {f.stem: f for f in Path(clean_dir).rglob("*.wav")}

        # Match noisy <-> clean by shared stem segment (DNS naming convention)
        # e.g. "noisy_fileid1234_snr10_..." <-> "clean_fileid1234_snr10_..."
        paired = []
        for stem, npath in noisy_files.items():
            key = stem.replace("noisy_", "").replace("noisy", "")
            for cstem, cpath in clean_files.items():
                ckey = cstem.replace("clean_", "").replace("clean", "")
                if key == ckey:
                    paired.append((npath, cpath))
                    break

        # Fallback: assume sorted order matches (DNS synthesizer guarantees this)
        if len(paired) == 0:
            nlist = sorted(noisy_files.values())
            clist = sorted(clean_files.values())
            assert len(nlist) == len(clist), \
                "Noisy and clean file counts don't match and names don't pair."
            paired = list(zip(nlist, clist))

        self.pairs       = paired
        self.sample_rate = sample_rate
        self.max_length  = int(sample_rate * max_duration_sec)
        self.stft        = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
        )
        print(f"DNS Dataset: {len(self.pairs)} pairs loaded.")

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        T = wav.shape[1]
        if T > self.max_length:
            start = random.randint(0, T - self.max_length)
            wav = wav[:, start:start + self.max_length]
        elif T < self.max_length:
            wav = torch.nn.functional.pad(wav, (0, self.max_length - T))
        return wav

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy = self._load(noisy_path)
        clean = self._load(clean_path)

        eps  = 1e-8
        peak = torch.max(torch.abs(noisy)) + eps
        noisy = noisy / peak
        clean = clean / peak

        mix_stft   = self.stft(noisy)
        clean_stft = self.stft(clean)

        norm = torch.max(torch.abs(mix_stft)) + eps
        return (mix_stft.real / norm, mix_stft.imag / norm,
                clean_stft.real / norm, clean_stft.imag / norm)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(clean_dir, noise_dir, rir_dir=None,
                    batch_size=8, snr_range=(0, 40),
                    val_split=0.1, num_workers=4, is_dns=False):
    """
    Returns (train_loader, val_loader) for selected dataset.
    If is_dns is True, uses DNSSpeechDataset (pre-mixed).
    Otherwise, uses ComplexSpeechDataset (synthetic on-the-fly mixing).
    """
    if is_dns:
        dataset = DNSSpeechDataset(noisy_dir=noise_dir, clean_dir=clean_dir)
    else:
        dataset = ComplexSpeechDataset(
            clean_dir, noise_dir, rir_dir=rir_dir, snr_range=snr_range
        )
    n_val   = max(1, int(val_split * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader


# Alias so torchaudio.functional.pad is available inside the class
import torch.nn.functional as _F
F_pad = _F.pad