import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio

from complex_model import DeepComplexUNet
from complex_data_prep import get_dataloaders

# --- CLUSTER PATHS ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "./complex_checkpoints_wind"
DATA_BASE = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/datasets_fullband"
CLEAN_DIR = os.path.join(DATA_BASE, "clean_fullband")
NOISE_DIR = os.path.join(DATA_BASE, "noise_fullband")
WIND_DIR = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/wind-noise-TL/synthetic_wind_dataset"
RIR_DIR = os.path.join(DATA_BASE, "impulse_responses")

BATCH_SIZE = 16
NUM_EPOCHS = 30 
LEARNING_RATE = 1e-5 # Stable Phase 2 rate
# loss function from the original paper
class wSDRLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super(wSDRLoss, self).__init__()
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop_length, normalized=True
        )
    # when call criterion(a,b,c,d,e,f) it will call this function

    # this recieve 6 tensors the real imag part of the mix, clean and predicted spectrograms
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
        
        eps = 1e-8 # tiny number used as a safelty buffer , add to denominator so that we can avoid dividing by zero and crash
        
        def sdr(target, pred):
            #  NUMERICAL STABILITY SHIELD
            num = torch.sum(target**2, dim=1) + eps
            den = torch.sum((target - pred)**2, dim=1) + eps
            # Clamp ratio to prevent log10(0) or log10(inf)
            ratio = torch.clamp(num / den, min=1e-7, max=1e7)
            return 10 * torch.log10(ratio)
            
        s_target = sdr(clean_wav, pred_wav) # how well speech is reconstructed
        n_target = sdr(noise_wav, pred_noise_wav) # how well is noise seperated
        
        clean_energy = torch.sum(clean_wav**2, dim=1) + eps
        noise_energy = torch.sum(noise_wav**2, dim=1) + eps
        # alpha is weight , if clean energy is high alpha is high focus on speech; if alpha is low noise is high concetrate on removing noise
        alpha = clean_energy / (clean_energy + noise_energy + eps)
        
        loss = -(alpha * s_target + (1 - alpha) * n_target)
            
        return torch.mean(loss)

class WindSpecialistCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.wsdr = wSDRLoss()

    def forward(self, mix_real, mix_imag, clean_real, clean_imag, pred_real, pred_imag):
        l_wsdr = self.wsdr(mix_real, mix_imag, clean_real, clean_imag, pred_real, pred_imag)
        
        #magnitude MSE
        eps_stab = 1e-7
        target_mag = torch.sqrt(clean_real**2 + clean_imag**2 + 1e-8)
        pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-8)
        #old was 0.2 and no phase correction
        l_mag = F.mse_loss(torch.pow(target_mag + eps_stab, 0.3), torch.pow(pred_mag + eps_stab, 0.3))
        
        #Phase Correction (Cosine Similarity)
        cos_sim = (pred_real * clean_real + pred_imag * clean_imag) / (pred_mag * target_mag + 1e-8)
        l_phase = torch.mean(1.0 - cos_sim)
        
        return l_wsdr + 1.0 * l_mag + 0.5 * l_phase

def freeze_encoder(model):
    print("Stable Phase 2: Freezing all Encoder layers (inc, down1-4).")
    for module in [model.inc, model.down1, model.down2, model.down3, model.down4]:
        for param in module.parameters():
            param.requires_grad = False

def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train() # tell pytorch we are in training mode now ;turn on batchnorm and dropout(well we are not using dropout)
    running_loss = 0.0 # initialise the running loss ; this is used to track average loss over the epoch
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for mix_real, mix_imag, clean_real, clean_imag in loop:
        mix_real, mix_imag = mix_real.to(DEVICE), mix_imag.to(DEVICE)
        clean_real, clean_imag = clean_real.to(DEVICE), clean_imag.to(DEVICE)
        
        mask_real, mask_imag = model(mix_real, mix_imag)
        pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
        pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real
        
        loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
        
        if torch.isnan(loss):
            continue
            
        optimizer.zero_grad() # clean the gradient from the previous iteration
        loss.backward() # calculate the gradient of the loss with respect to model params
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item() # add the loss to the running loss
        loop.set_description(f"Transfer Learning Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
        
    return running_loss / (len(dataloader) + 1e-8) # eps to prevent div-by-zero
 
# (Rest of validation and main remain standard)
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    count = 0
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
    print(f"starting wind transfer learning")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # initialize Model
    model = DeepComplexUNet(n_channels=1).to(DEVICE)
    
    #load pre trained weights
    PRETRAINED_PATH = "../complex_checkpoints/dcunet_epoch_120.pth" 
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading weights from: {PRETRAINED_PATH}")
        checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("weights not found")

    #freeze encoder
    freeze_encoder(model)
    
    #optimizer for trainable parameters only
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    criterion = WindSpecialistCriterion()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    #load data- 10 % gnrl noise, 90 % wind noise
    train_loader, val_loader = get_dataloaders(
        CLEAN_DIR, WIND_DIR, NOISE_DIR, RIR_DIR, 
        batch_size=BATCH_SIZE, snr_range=(-10, 5), max_files=10000
    )
    
    log_file = os.path.join(SAVE_DIR, "wind_transfer_log.csv")
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss Train: {avg_loss:.4f} | Val: {val_loss:.4f}")
        
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if epoch == 0: writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])
            writer.writerow([epoch + 1, avg_loss, val_loss])
            
        scheduler.step(val_loss)
        
        # Save checkpoints
        save_path = os.path.join(SAVE_DIR, "wind_specialist_latest.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss}, save_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "wind_specialist_best.pth"))

if __name__ == "__main__":
    main()
