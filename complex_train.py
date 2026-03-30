import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchaudio

from complex_model import DeepComplexUNet
from complex_data_prep import get_dataloaders

class wSDRLoss(nn.Module):
    """
    Weighted Source-to-Distortion Ratio (wSDR) loss function
    evaluated directly on the complex spectrogram (Parseval's theorem).
    """
    def __init__(self, n_fft=512, hop_length=256):
        super(wSDRLoss, self).__init__()
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, 
            hop_length=hop_length, 
            normalized=True
        ).to(DEVICE) # Assuming DEVICE is globally defined

    def forward(self, mix_real, mix_imag, clean_real, clean_imag, pred_real, pred_imag):
        # 1. Apply ISTFT to convert frequency predictions back into raw time-domain waveforms
        mix_complex = torch.complex(mix_real, mix_imag)
        clean_complex = torch.complex(clean_real, clean_imag)
        pred_complex = torch.complex(pred_real, pred_imag)
        
        mix_wav = self.istft(mix_complex)
        clean_wav = self.istft(clean_complex)
        pred_wav = self.istft(pred_complex)
        
        # 2. Extract Noise waveforms
        noise_wav = mix_wav - clean_wav
        pred_noise_wav = mix_wav - pred_wav
        
        # Flatten
        clean_wav = clean_wav.flatten(1)
        pred_wav = pred_wav.flatten(1)
        noise_wav = noise_wav.flatten(1)
        pred_noise_wav = pred_noise_wav.flatten(1)
        
        eps = 1e-8
        
        # 3. Energy-ratio SDR (True formula as defined in paper)
        # SDR = ||target||^2 / ||target - pred||^2
        def sdr(target, pred):
            num = torch.sum(target**2, dim=1)
            den = torch.sum((target - pred)**2, dim=1)
            # The paper formula operates natively on this ratio rather than cosine similarity
            return num / (den + eps)
            
        s_target = sdr(clean_wav, pred_wav)
        n_target = sdr(noise_wav, pred_noise_wav)
        
        # 4. Energy weighting alpha
        clean_energy = torch.sum(clean_wav**2, dim=1)
        noise_energy = torch.sum(noise_wav**2, dim=1)
        alpha = clean_energy / (clean_energy + noise_energy + eps)
        
        # Negative because we want to maximize wSDR correlation
        loss = - (alpha * s_target + (1 - alpha) * n_target)
        return torch.mean(loss)


CLEAN_DIR = "./MS-SNSD/clean_train"
NOISE_DIR = "./MS-SNSD/noise_train"
BATCH_SIZE = 16
# NUM_EPOCHS = 100 # Increased epochs because Phase is harder to learn
NUM_EPOCHS = 120 # Fine-tuning for 20 more epochs with Aggressive SNR
LEARNING_RATE = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "./complex_checkpoints"

def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    # noisy audio input and ground truth clean audio
    for mix_real, mix_imag, clean_real, clean_imag in loop:
        # Move everything to GPU
        mix_real = mix_real.to(DEVICE)
        mix_imag = mix_imag.to(DEVICE)
        clean_real = clean_real.to(DEVICE)
        clean_imag = clean_imag.to(DEVICE)
        
        # In a typical Complex Masking paper, the model outputs a Complex Mask (cRM)
        # We multiply the cRM by the Mixed Input to estimate the Clean output.
        mask_real, mask_imag = model(mix_real, mix_imag)
        
        # Apply Complex Ratio Mask: (Mix_r + i*Mix_i) * (Mask_r + i*Mask_i)
        # S_r = M_r * X_r - M_i * X_i
        # S_i = M_r * X_i + M_i * X_r
        pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
        pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real
        
        # Calculate wSDR Loss
        loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        # print live loss per batch
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for mix_real, mix_imag, clean_real, clean_imag in dataloader:
            mix_real = mix_real.to(DEVICE)
            mix_imag = mix_imag.to(DEVICE)
            clean_real = clean_real.to(DEVICE)
            clean_imag = clean_imag.to(DEVICE)
            
            mask_real, mask_imag = model(mix_real, mix_imag)
            
            pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
            pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real
            
            loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
            running_loss += loss.item()

    return running_loss / len(dataloader)

def main():
    print(f"Training DCUNet on device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Find the latest checkpoint if it exists
    start_epoch = 0
    checkpoint_path = None
    if os.path.exists(SAVE_DIR):
        # Only look for numbered periodic checkpoints (ignoring best_dcunet.pth)
        checkpoints = [f for f in os.listdir(SAVE_DIR) if f.startswith('dcunet_epoch_') and f.endswith('.pth')]
        if checkpoints:
            # Sort by epoch number: dcunet_epoch_40.pth -> 40
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(SAVE_DIR, latest_checkpoint)
    
    model = DeepComplexUNet(n_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 2. Load checkpoint if found
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Restarting from Epoch {start_epoch}")

    print("Loading Complex Spectral Data (Aggressive SNR)...")
    # train_loader, val_loader = get_dataloaders(CLEAN_DIR, NOISE_DIR, batch_size=BATCH_SIZE)
    train_loader, val_loader = get_dataloaders(CLEAN_DIR, NOISE_DIR, batch_size=BATCH_SIZE, snr_range=(-15, 10))
    
    # We use wSDRLoss (Weighted Source-to-Distortion Ratio). 
    # It directly measures complex temporal/spectral geometry rather than raw MSE pixel loss.
    criterion = wSDRLoss()
    
    # Optional but highly recommended: Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_path = os.path.join(SAVE_DIR, "best_dcunet.pth")
    if os.path.exists(best_path):
        try:
            best_ckpt = torch.load(best_path, map_location=DEVICE)
            # Resume tracking from previous best validation score so we don't accidentally overwrite a better older model
            best_val_loss = best_ckpt.get('val_loss', float('inf')) 
            print(f"Resuming tracking with previous best validation loss: {best_val_loss:.6f}")
        except:
            pass

    log_file = os.path.join(SAVE_DIR, "training_log.csv")
    if not os.path.exists(log_file) and start_epoch == 0:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_wSDR_Loss", "Val_wSDR_Loss"])

    print("Starting Deep Complex Training...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. wSDR Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Log to CSV natively so we can visually check for Overfitting later
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, val_loss])
        
        # Update Scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Track and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(SAVE_DIR, "best_dcunet.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_path)
            print(f"New best model saved to {best_path} (Val Loss: {best_val_loss:.6f})")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"dcunet_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss
            }, save_path)
            print(f"Saved periodic complex checkpoint to {save_path}")

if __name__ == "__main__":
    main()
