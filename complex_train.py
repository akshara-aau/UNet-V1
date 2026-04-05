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

# --- 🛰️ TARGET CLUSTER CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "./complex_checkpoints"

DATA_BASE = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/datasets_fullband"
CLEAN_DIR = os.path.join(DATA_BASE, "clean_fullband")
NOISE_DIR = os.path.join(DATA_BASE, "noise_fullband")
RIR_DIR = os.path.join(DATA_BASE, "impulse_responses")

BATCH_SIZE = 16
NUM_EPOCHS = 120 
LEARNING_RATE = 2e-4

class wSDRLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super(wSDRLoss, self).__init__()
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
            # 🛡️ NUMERICAL STABILITY SHIELD
            num = torch.sum(target**2, dim=1) + eps
            den = torch.sum((target - pred)**2, dim=1) + eps
            # Clamp ratio to prevent log10(0) or log10(inf)
            ratio = torch.clamp(num / den, min=1e-7, max=1e7)
            return 10 * torch.log10(ratio)
            
        s_target = sdr(clean_wav, pred_wav)
        n_target = sdr(noise_wav, pred_noise_wav)
        
        clean_energy = torch.sum(clean_wav**2, dim=1) + eps
        noise_energy = torch.sum(noise_wav**2, dim=1) + eps
        alpha = clean_energy / (clean_energy + noise_energy + eps)
        
        loss = - (alpha * s_target + (1 - alpha) * n_target)
        
        # 🧪 FINAL SHIELD: Filter out any NaNs that managed to break through
        loss = loss[~torch.isnan(loss)]
        if loss.numel() == 0:
            return torch.tensor(0.0, device=mix_real.device, requires_grad=True)
            
        return torch.mean(loss)

def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for mix_real, mix_imag, clean_real, clean_imag in loop:
        mix_real, mix_imag = mix_real.to(DEVICE), mix_imag.to(DEVICE)
        clean_real, clean_imag = clean_real.to(DEVICE), clean_imag.to(DEVICE)
        
        mask_real, mask_imag = model(mix_real, mix_imag)
        pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
        pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real
        
        loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
        
        # 🚀 If loss is still NaN, skip this batch entirely
        if torch.isnan(loss) or loss.item() == 0:
            continue
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
        
    return running_loss / (len(dataloader) + 1e-8) # eps to prevent div-by-zero

# (Rest of validation and main remain standard)
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for mix_real, mix_imag, clean_real, clean_imag in dataloader:
            mix_real, mix_imag = mix_real.to(DEVICE), mix_imag.to(DEVICE)
            clean_real, clean_imag = clean_real.to(DEVICE), clean_imag.to(DEVICE)
            mask_real, mask_imag = model(mix_real, mix_imag)
            pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
            pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real
            loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, filename)

def main():
    print(f"📡 DNS TRAINING: Using full dataset from: {DATA_BASE}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    start_epoch = 0
    checkpoint_path = None
    if os.path.exists(SAVE_DIR):
        checkpoints = [f for f in os.listdir(SAVE_DIR) if f.startswith('dcunet_epoch_') and f.endswith('.pth')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(SAVE_DIR, checkpoints[-1])
    
    model = DeepComplexUNet(n_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = wSDRLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print("🛰️ Initializing Dynamic Real-World Data Pipeline...")
    train_loader, val_loader = get_dataloaders(CLEAN_DIR, NOISE_DIR, RIR_DIR, batch_size=BATCH_SIZE, snr_range=(-15, 10))
    
    log_file = os.path.join(SAVE_DIR, "training_log.csv")
    if not os.path.exists(log_file) and start_epoch == 0:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_wSDR_Loss", "Val_wSDR_Loss"])

    for epoch in range(start_epoch, NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] wSDR Train: {avg_loss:.4f} | Val: {val_loss:.4f}")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, val_loss])
        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"dcunet_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_path)
            print(f"Checkpoint saved: {save_path}")

if __name__ == "__main__":
    main()
