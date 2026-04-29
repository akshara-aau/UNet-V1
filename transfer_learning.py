import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from audio_segmentation import segment_audio
from complex_data_prep import get_dataloaders
from complex_model import DeepComplexUNet


def get_device(device_pref: str) -> torch.device:
    device_pref = (device_pref or "auto").lower()
    if device_pref == "cuda":
        return torch.device("cuda")
    if device_pref == "mps":
        return torch.device("cpu")
    if device_pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("cpu")


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def configure_transfer_learning(
    model: DeepComplexUNet,
    *,
    freeze_encoder: bool,
    freeze_decoder: bool,
    freeze_output: bool,
    unfreeze_bn: bool,
) -> None:
    # Encoder
    if freeze_encoder:
        for m in [model.inc, model.down1, model.down2, model.down3, model.down4]:
            set_trainable(m, False)

    # Decoder
    if freeze_decoder:
        for m in [model.up1, model.up2, model.up3, model.up4]:
            set_trainable(m, False)

    # Output layer
    if freeze_output:
        set_trainable(model.out_conv, False)

    # Optional: keep BatchNorm trainable for domain adaptation
    if unfreeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d,)):
                set_trainable(m, True)


class wSDRLoss(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 256):
        super().__init__()
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop_length, normalized=True
        )

    def forward(self, mix_real, mix_imag, clean_real, clean_imag, pred_real, pred_imag):
        self.istft = self.istft.to(mix_real.device)
        mix_complex = torch.complex(mix_real, mix_imag)
        clean_complex = torch.complex(clean_real, clean_imag)
        pred_complex = torch.complex(pred_real, pred_imag)

        mix_wav = self.istft(mix_complex)
        clean_wav = self.istft(clean_complex)
        pred_wav = self.istft(pred_complex)

        noise_wav = mix_wav - clean_wav
        pred_noise_wav = mix_wav - pred_wav

        clean_wav = clean_wav.flatten(1)
        pred_wav = pred_wav.flatten(1)
        noise_wav = noise_wav.flatten(1)
        pred_noise_wav = pred_noise_wav.flatten(1)

        eps = 1e-8

        def sdr(target, pred):
            num = torch.sum(target**2, dim=1) + eps
            den = torch.sum((target - pred) ** 2, dim=1) + eps
            ratio = torch.clamp(num / den, min=1e-7, max=1e7)
            return 10 * torch.log10(ratio)

        s_target = sdr(clean_wav, pred_wav)
        n_target = sdr(noise_wav, pred_noise_wav)

        clean_energy = torch.sum(clean_wav**2, dim=1) + eps
        noise_energy = torch.sum(noise_wav**2, dim=1) + eps
        alpha = clean_energy / (clean_energy + noise_energy + eps)

        loss = - (alpha * s_target + (1 - alpha) * n_target)
        loss = loss[~torch.isnan(loss)]
        if loss.numel() == 0:
            return torch.tensor(0.0, device=mix_real.device, requires_grad=True)
        return torch.mean(loss)


@dataclass
class TrainConfig:
    clean_dir: str
    noise_dir: str
    save_dir: str
    pretrained: str
    device: torch.device
    batch_size: int
    epochs: int
    lr: float
    snr_min: float
    snr_max: float


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for mix_real, mix_imag, clean_real, clean_imag in loop:
        mix_real, mix_imag = mix_real.to(device), mix_imag.to(device)
        clean_real, clean_imag = clean_real.to(device), clean_imag.to(device)

        mask_real, mask_imag = model(mix_real, mix_imag)
        pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
        pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real

        loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
        if torch.isnan(loss) or loss.item() == 0:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    return running_loss / (len(dataloader) + 1e-8)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for mix_real, mix_imag, clean_real, clean_imag in dataloader:
            mix_real, mix_imag = mix_real.to(device), mix_imag.to(device)
            clean_real, clean_imag = clean_real.to(device), clean_imag.to(device)

            mask_real, mask_imag = model(mix_real, mix_imag)
            pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
            pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real

            loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
            running_loss += loss.item()
    return running_loss / max(1, len(dataloader))


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, filename)


def main():
    parser = argparse.ArgumentParser(description="Transfer learning / fine-tuning for DeepComplexUNet.")

    parser.add_argument("--pretrained", default="dcunet_epoch_120_DND.pth", help="Path to pretrained .pth file")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    parser.add_argument("--clean_dir", required=True, help="Directory containing clean wav files")
    parser.add_argument("--noise_dir", required=True, help="Directory containing noise wav files")

    parser.add_argument("--save_dir", default="./transfer_checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--snr_min", type=float, default=-15.0)
    parser.add_argument("--snr_max", type=float, default=10.0)

    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder blocks (inc, down1..down4)")
    parser.add_argument("--freeze_decoder", action="store_true", help="Freeze decoder blocks (up1..up4)")
    parser.add_argument("--freeze_output", action="store_true", help="Freeze output conv (out_conv)")
    parser.add_argument("--unfreeze_bn", action="store_true", help="Keep BatchNorm trainable even if frozen")

    args = parser.parse_args()

    print(f"cwd: {os.getcwd()}")
    print(f"clean_dir: {args.clean_dir}")
    print(f"noise_dir: {args.noise_dir}")

    clean_path = Path(args.clean_dir).expanduser()
    noise_path = Path(args.noise_dir).expanduser()
    print(f"clean_dir_resolved: {clean_path.resolve() if clean_path.exists() else clean_path}")
    print(f"noise_dir_resolved: {noise_path.resolve() if noise_path.exists() else noise_path}")

    if not clean_path.exists():
        raise FileNotFoundError(f"--clean_dir does not exist: {clean_path}")
    if not noise_path.exists():
        raise FileNotFoundError(f"--noise_dir does not exist: {noise_path}")

    clean_wavs = list(clean_path.rglob("*.wav"))
    noise_wavs = list(noise_path.rglob("*.wav"))
    print(f"clean_wav_count: {len(clean_wavs)}")
    print(f"noise_wav_count: {len(noise_wavs)}")
    if len(clean_wavs) == 0:
        raise RuntimeError(f"No .wav files found under --clean_dir: {clean_path}")
    if len(noise_wavs) == 0:
        raise RuntimeError(f"No .wav files found under --noise_dir: {noise_path}")

    device = get_device(args.device)
    cfg = TrainConfig(
        clean_dir=str(clean_path),
        noise_dir=str(noise_path),
        save_dir=args.save_dir,
        pretrained=args.pretrained,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
    )

    print(f"Using device: {cfg.device}")
    os.makedirs(cfg.save_dir, exist_ok=True)

    model = DeepComplexUNet(n_channels=1).to(cfg.device)
    load_checkpoint_into_model(model, cfg.pretrained, cfg.device)
    print(f"Loaded pretrained weights from: {cfg.pretrained}")

    configure_transfer_learning(
        model,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        freeze_output=args.freeze_output,
        unfreeze_bn=args.unfreeze_bn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters left. Remove some --freeze_* flags.")

    optimizer = optim.Adam(trainable_params, lr=cfg.lr)
    criterion = wSDRLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    train_loader, val_loader = get_dataloaders(
        cfg.clean_dir,
        cfg.noise_dir,
        batch_size=cfg.batch_size,
        snr_range=(cfg.snr_min, cfg.snr_max),
    )

    log_file = os.path.join(cfg.save_dir, "transfer_training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_wSDR_Loss", "Val_wSDR_Loss", "LR"])

    best_val = float("inf")
    for epoch in range(cfg.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch + 1}/{cfg.epochs}] LR: {current_lr:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device, epoch, cfg.epochs)
        val_loss = validate_one_epoch(model, val_loader, criterion, cfg.device)
        print(f"wSDR Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])

        scheduler.step(val_loss)

        latest_path = os.path.join(cfg.save_dir, "latest_checkpoint.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, best_val, latest_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(cfg.save_dir, "best_checkpoint.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val, best_path)

        if (epoch + 1) % 10 == 0:
            arch_path = os.path.join(cfg.save_dir, f"transfer_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val, arch_path)


if __name__ == "__main__":
    # segment_audio('dynamic-20.wav', 'my_data/clean', segment_duration=4, overlap=0.0, pad_last=True, skip_silence=False, silence_threshold=0.01)
    # segment_audio('dynamic-20-noise.wav', 'my_data/noise', segment_duration=4, overlap=0.0, pad_last=True, skip_silence=False, silence_threshold=0.01)

    main()

