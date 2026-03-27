import os
import torch
import torch.optim as optim
import torchaudio
import numpy as np
from tqdm import tqdm

from complex_model    import DeepComplexUNet
from complex_data_prep import get_dataloaders
from complex_losses   import CombinedLoss

# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------

# Data directories
# CLEAN_DIR = "./MS-SNSD/clean_train"
# NOISE_DIR = "./MS-SNSD/noise_train"

# DNS Challenge directories
CLEAN_DIR = "./DNS-Challenge/datasets/clean/"
NOISE_DIR = "./DNS-Challenge/datasets/noisy/"
RIR_DIR   = None        # set to path of RIR .wav files, or None to skip
USE_DNS   = True        # True if using pre-mixed DNS dataset

# STFT settings — must match inference
N_FFT      = 512
HOP_LENGTH = 256

# Model
MODEL_SIZE = 'dcu16'    # 'dcu8' | 'dcu16' | 'dcu20'

# Training
BATCH_SIZE  = 8         # reduce to 4 if OOM; 16 is fine for dcu8
NUM_EPOCHS  = 150
LR          = 1e-3
SNR_RANGE   = (0, 40)   # DNS-style wide range; use (-5, 15) for MS-SNSD only

# Paths
SAVE_DIR        = "./checkpoints"
BEST_MODEL_PATH = "./best_model.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Evaluation with PESQ + STOI
# ---------------------------------------------------------------------------

def evaluate(model, val_loader, istft, device, max_batches=40):
    """
    Returns (mean_PESQ, mean_STOI) over up to max_batches validation batches.
    Requires:  pip install pesq pystoi
    """
    try:
        from pesq   import pesq   as pesq_fn
        from pystoi import stoi   as stoi_fn
    except ImportError:
        print("  [eval] Install 'pesq' and 'pystoi' for metrics: "
              "pip install pesq pystoi")
        return None, None

    model.eval()
    pesq_scores, stoi_scores = [], []

    with torch.no_grad():
        for batch_idx, (mix_r, mix_i, clean_r, clean_i) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            mix_r, mix_i = mix_r.to(device), mix_i.to(device)
            clean_r, clean_i = clean_r.to(device), clean_i.to(device)

            mask_r, mask_i = model(mix_r, mix_i)

            # Apply complex ratio mask
            est_r = mask_r * mix_r - mask_i * mix_i
            est_i = mask_r * mix_i + mask_i * mix_r

            # Convert to waveforms
            est_wav   = istft(torch.complex(est_r.squeeze(1),   est_i.squeeze(1))).cpu().numpy()
            clean_wav = istft(torch.complex(clean_r.squeeze(1), clean_i.squeeze(1))).cpu().numpy()

            for j in range(est_wav.shape[0]):
                e = est_wav[j]
                c = clean_wav[j]

                # Normalise before scoring (required by pesq/stoi)
                e = e / (np.max(np.abs(e)) + 1e-8)
                c = c / (np.max(np.abs(c)) + 1e-8)

                try:
                    pesq_scores.append(pesq_fn(16000, c, e, 'wb'))
                    stoi_scores.append(stoi_fn(c, e, 16000, extended=False))
                except Exception:
                    pass    # skip clips that are too short / silent

    if len(pesq_scores) == 0:
        return None, None

    return float(np.mean(pesq_scores)), float(np.mean(stoi_scores))


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler,
                    criterion, istft, epoch, num_epochs, device):
    model.train()
    running_loss = 0.0
    running_wsdr = 0.0
    running_mr   = 0.0

    loop = tqdm(dataloader, total=len(dataloader), leave=False)

    for mix_real, mix_imag, clean_real, clean_imag in loop:
        mix_real  = mix_real.to(device)
        mix_imag  = mix_imag.to(device)
        clean_real = clean_real.to(device)
        clean_imag = clean_imag.to(device)

        # ---- Forward pass ----
        mask_r, mask_i = model(mix_real, mix_imag)

        # Apply Complex Ratio Mask:  est = mask * noisy  (complex multiply)
        #   (M_r + j*M_i)(Y_r + j*Y_i) = (M_r*Y_r - M_i*Y_i) + j(M_r*Y_i + M_i*Y_r)
        est_real = mask_r * mix_real  - mask_i * mix_imag
        est_imag = mask_r * mix_imag  + mask_i * mix_real

        # Convert estimated and targets to waveforms for time-domain loss
        est_wav   = istft(torch.complex(est_real.squeeze(1),   est_imag.squeeze(1)))
        clean_wav = istft(torch.complex(clean_real.squeeze(1), clean_imag.squeeze(1)))
        noisy_wav = istft(torch.complex(mix_real.squeeze(1),   mix_imag.squeeze(1)))

        # ---- Combined wSDR + Multi-Resolution STFT loss ----
        loss, loss_wsdr, loss_mr = criterion(clean_wav, noisy_wav, est_wav)

        # ---- Backward pass ----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()   # OneCycleLR steps every batch

        running_loss += loss.item()
        running_wsdr += loss_wsdr.item()
        running_mr   += loss_mr.item()

        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            wsdr=f"{loss_wsdr.item():.4f}",
            mr=f"{loss_mr.item():.4f}"
        )

    n = len(dataloader)
    return running_loss / n, running_wsdr / n, running_mr / n


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    print(f"Training DCUNet ({MODEL_SIZE}) on {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Data ----
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        CLEAN_DIR, NOISE_DIR,
        rir_dir=RIR_DIR,
        batch_size=BATCH_SIZE,
        snr_range=SNR_RANGE,
        is_dns=USE_DNS
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = DeepComplexUNet(n_channels=1, model_size=MODEL_SIZE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # ---- ISTFT (used for time-domain loss and evaluation) ----
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True
    ).to(DEVICE)

    # ---- Loss, Optimizer, Scheduler ----
    criterion = CombinedLoss(mr_weight=0.5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    # OneCycleLR: linear warmup for 5%, cosine decay for the rest.
    # Much better convergence than ReduceLROnPlateau for encoder-decoders.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.05,       # 5% warmup
        anneal_strategy='cos',
        div_factor=25,        # initial lr = max_lr / 25
        final_div_factor=1e4  # final lr = initial_lr / 1e4
    )

    # ---- Resume from checkpoint if one exists ----
    start_epoch  = 0
    best_pesq    = -999.0

    checkpoints = sorted(
        [f for f in os.listdir(SAVE_DIR) if f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    if checkpoints:
        ckpt_path = os.path.join(SAVE_DIR, checkpoints[-1])
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_pesq   = ckpt.get('best_pesq', -999.0)
        print(f"  Resuming from epoch {start_epoch}, best PESQ so far: {best_pesq:.3f}")

    # ---- Training loop ----
    print("\nStarting training...")
    for epoch in range(start_epoch, NUM_EPOCHS):

        avg_loss, avg_wsdr, avg_mr = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, istft, epoch, NUM_EPOCHS, DEVICE
        )

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"loss={avg_loss:.4f}  wsdr={avg_wsdr:.4f}  mr={avg_mr:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # ---- Evaluate every 5 epochs ----
        if (epoch + 1) % 5 == 0:
            pesq_score, stoi_score = evaluate(
                model, val_loader, istft, DEVICE
            )
            if pesq_score is not None:
                print(f"  [Val] PESQ={pesq_score:.3f}  STOI={stoi_score:.3f}")

                # Save best model by PESQ (not by fixed epoch schedule)
                if pesq_score > best_pesq:
                    best_pesq = pesq_score
                    torch.save({
                        'epoch':               epoch,
                        'model_size':          MODEL_SIZE,
                        'model_state_dict':    model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'pesq':                pesq_score,
                        'stoi':                stoi_score,
                        'best_pesq':           best_pesq,
                    }, BEST_MODEL_PATH)
                    print(f"  ✓ New best PESQ {pesq_score:.3f} — saved to {BEST_MODEL_PATH}")

        # ---- Periodic checkpoint every 10 epochs ----
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"dcunet_epoch_{epoch+1}.pth")
            torch.save({
                'epoch':                epoch,
                'model_size':           MODEL_SIZE,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 avg_loss,
                'best_pesq':            best_pesq,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\nTraining complete. Best PESQ: {best_pesq:.3f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()