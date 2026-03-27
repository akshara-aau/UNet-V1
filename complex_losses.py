import torch
import torch.nn as nn
import torch.nn.functional as F


def wsdr_loss(clean_wav, noisy_wav, estimated_wav, eps=1e-8):
    """
    Weighted Signal-to-Distortion Ratio loss from the DCUNet paper.

    All inputs are [Batch, Time] waveforms (after ISTFT).

    The loss has two terms:
      - SDR between clean target and estimated output
      - SDR between the noise component and residual noise in the output
    They are weighted by alpha = (speech energy) / (speech + noise energy),
    which adapts automatically: noisier samples penalise noise residuals more.

    We use (1 - SDR_ratio) instead of -log(SDR) for numerical stability.
    """
    noise     = noisy_wav - clean_wav          # actual noise in the mixture
    noise_est = estimated_wav - clean_wav      # residual noise in the output

    def sdr_ratio(ref, est):
        # Numerator:   dot product squared
        # Denominator: product of energies
        num = (ref * est).sum(dim=-1).pow(2)
        den = ref.pow(2).sum(dim=-1) * est.pow(2).sum(dim=-1) + eps
        return num / den

    # alpha: fraction of the mixture that is speech
    clean_energy = clean_wav.pow(2).sum(dim=-1)
    noise_energy = noise.pow(2).sum(dim=-1)
    alpha = clean_energy / (clean_energy + noise_energy + eps)

    loss_speech = 1.0 - sdr_ratio(clean_wav, estimated_wav)
    loss_noise  = 1.0 - sdr_ratio(noise, noise_est)

    loss = alpha * loss_speech + (1.0 - alpha) * loss_noise
    return loss.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss using three window sizes simultaneously.

    Short windows (512)  capture transients and consonants ("s", "f", "t").
    Medium windows (1024) balance time and frequency resolution.
    Long windows (2048)  capture tonal noise patterns and pitch.

    Each scale contributes:
      - Spectral convergence loss (Frobenius norm ratio)
      - Log-magnitude MSE

    Total loss is the mean across all scales.
    """

    def __init__(self,
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[128, 256,  512],
                 win_lengths=[512, 1024, 2048]):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes   = fft_sizes
        self.hop_sizes   = hop_sizes
        self.win_lengths = win_lengths

    def _stft(self, x, fft_size, hop_size, win_length):
        window = torch.hann_window(win_length, device=x.device)
        return torch.stft(x, fft_size, hop_size, win_length,
                          window=window, return_complex=True)

    def _single_scale_loss(self, estimated, clean, fft_size, hop_size, win_length):
        X = self._stft(estimated, fft_size, hop_size, win_length)
        Y = self._stft(clean,     fft_size, hop_size, win_length)

        X_mag = torch.abs(X) + 1e-8
        Y_mag = torch.abs(Y) + 1e-8

        # Spectral convergence: ||Y_mag - X_mag||_F / ||Y_mag||_F
        sc_loss = torch.norm(Y_mag - X_mag, p='fro') / \
                  (torch.norm(Y_mag, p='fro') + 1e-8)

        # Log magnitude MSE
        lm_loss = F.mse_loss(torch.log(X_mag), torch.log(Y_mag))

        return sc_loss + lm_loss

    def forward(self, estimated, clean):
        """
        estimated, clean: [Batch, Time] waveforms
        Returns scalar loss.
        """
        total = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            total += self._single_scale_loss(estimated, clean, fft, hop, win)
        return total / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """
    wSDR (time-domain) + weighted MultiResolutionSTFTLoss.

    mr_weight controls the balance. 0.5 is a good starting point — it
    lets wSDR dominate (preserving the paper's main contribution) while
    the STFT loss improves high-frequency detail.
    """

    def __init__(self, mr_weight=0.5):
        super().__init__()
        self.mr_loss  = MultiResolutionSTFTLoss()
        self.mr_weight = mr_weight

    def forward(self, clean_wav, noisy_wav, estimated_wav):
        loss_wsdr = wsdr_loss(clean_wav, noisy_wav, estimated_wav)
        loss_mr   = self.mr_loss(estimated_wav, clean_wav)
        return loss_wsdr + self.mr_weight * loss_mr, loss_wsdr, loss_mr