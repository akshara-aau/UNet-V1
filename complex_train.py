import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from complex_model import DeepComplexUNet
from complex_data_prep import get_dataloaders

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
        # However, for simplicity and stability, we will just have the model directly guess
        # the normalized clean Real and Imaginary spectrograms based on the mixture.
        
        # Forward Pass: Reconstruct Real and Imaginary Clean Speech
        pred_clean_real, pred_clean_imag = model(mix_real, mix_imag)
        
        # Calculate Separate Losses
        loss_real = criterion(pred_clean_real, clean_real)
        loss_imag = criterion(pred_clean_imag, clean_imag)
        
        # The Complex Loss is simply Re + Im difference
        loss = loss_real + loss_imag
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        # print live loss per batch
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)

def main():
    print(f"Training DCUNet on device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Find the latest checkpoint if it exists
    start_epoch = 0
    checkpoint_path = None
    if os.path.exists(SAVE_DIR):
        checkpoints = [f for f in os.listdir(SAVE_DIR) if f.endswith('.pth')]
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
    
    # We use MSELoss. Because of the Sigmoid/Tanh constraints on inputs, 
    # MSE operates perfectly to calculate pixel-by-pixel real/imag differences.
    criterion = nn.MSELoss() 
    
    # Optional but highly recommended: Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    print("Starting Deep Complex Training...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Complex MSE Loss: {avg_loss:.6f}")
        
        # Update Scheduler based on training loss (ideally use validation loss if added)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"dcunet_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved complex checkpoint to {save_path}")

if __name__ == "__main__":
    main()
