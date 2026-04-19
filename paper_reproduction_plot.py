import os
# Fix for Mac stability
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from complex_model import DeepComplexUNet
# --- CONFIGURATION ---
MODEL_PATH = "./complex_checkpoints/dcunet_epoch_120.pth"
TEST_NOISY_DIR = "./voicebank_wav/noisy"
TEST_CLEAN_DIR = "./voicebank_wav/clean"
N_FFT = 512
HOP_LENGTH = 256
MAX_FILES = 20  # Look at 20 files to get a dense cloud
POINTS_PER_FILE = 2000
def get_stft(waveform):
    waveform = torch.from_numpy(waveform).float()
    if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
    # Matching the model's normalisation strategy
    max_amp = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / max_amp
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                      window=torch.hann_window(N_FFT).to(waveform.device), 
                      return_complex=True, normalized=True)
    return stft

def run_reproduction():
    print(f"  Starting Aggregated Phase Analysis (Reproduction Mode)")
    device = torch.device('cpu')
    model = DeepComplexUNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    files = [f for f in os.listdir(TEST_NOISY_DIR) if f.endswith(".wav")]
    files = files[:MAX_FILES]
    all_ideal_real = []
    all_ideal_imag = []
    all_pred_real = []
    all_pred_imag = []
    
    print(f" Gathering data from {len(files)} files")
    for f_name in files:
        noisy_np, _ = sf.read(os.path.join(TEST_NOISY_DIR, f_name))
        clean_np, _ = sf.read(os.path.join(TEST_CLEAN_DIR, f_name))
        noisy_stft = get_stft(noisy_np)
        clean_stft = get_stft(clean_np)
        # 1. Calculate Ideal CRM
        eps = 1e-8
        ideal_crm = clean_stft / (noisy_stft + eps)
        # 2. Get Predicted CRM
        with torch.no_grad():
            norm_factor = torch.max(torch.abs(noisy_stft)) + eps
            real_in = (noisy_stft.real / norm_factor).unsqueeze(1)
            imag_in = (noisy_stft.imag / norm_factor).unsqueeze(1)
            mask_real, mask_imag = model(real_in, imag_in)
        # 3. Sample Points
        ir = ideal_crm.real.numpy().flatten()
        ii = ideal_crm.imag.numpy().flatten()
        pr = mask_real.numpy().flatten()
        pi = mask_imag.numpy().flatten()
        # Filtering for reasonable range
        mask = (np.abs(ir) < 2) & (np.abs(ii) < 2)
        ir, ii, pr, pi = ir[mask], ii[mask], pr[mask], pi[mask]
        idx = np.random.choice(len(ir), min(POINTS_PER_FILE, len(ir)), replace=False)
        all_ideal_real.extend(ir[idx])
        all_ideal_imag.extend(ii[idx])
        all_pred_real.extend(pr[idx])
        all_pred_imag.extend(pi[idx])

    # --- PLOTTING (Exactly like the Paper) ---
    print(" Creating High-Fidelity Reproduction Plot")
    plt.figure(figsize=(10, 5))
    # Ground Truth
    plt.subplot(1, 2, 1)
    plt.scatter(all_ideal_real, all_ideal_imag, alpha=0.1, s=1, color='navy')
    plt.axhline(0, color='black', lw=0.5, alpha=0.5)
    plt.axvline(0, color='black', lw=0.5, alpha=0.5)
    # Drawing the unit circle like in the paper
    circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', lw=1)
    plt.gca().add_patch(circle)
    plt.title("Ground Truth (Ideal Mask)", fontsize=12)
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.xlim(-2, 2); plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Ours
    plt.subplot(1, 2, 2)
    plt.scatter(all_pred_real, all_pred_imag, alpha=0.1, s=1, color='purple')
    plt.axhline(0, color='black', lw=0.5, alpha=0.5)
    plt.axvline(0, color='black', lw=0.5, alpha=0.5)
    circle2 = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', lw=1)
    plt.gca().add_patch(circle2)
    plt.title("Ours (Tanh + wSDR Prediction)", fontsize=12)
    plt.xlabel("Real Part")
    plt.xlim(-2, 2); plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.suptitle("Comparative Complex Distribution (Aggregate Test Set)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    res_path = "exact_paper_reproduction.png"
    plt.savefig(res_path, dpi=300)
    print(f" Success! Plot saved as: {res_path}")
    plt.show()

if __name__ == "__main__":
    run_reproduction()
