import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from complex_model import DeepComplexUNet

# --- CONFIGURATION ---
MODEL_PATH = "./complex_checkpoints/dcunet_epoch_120.pth"
SAMPLE_NOISY = "./voicebank_wav/noisy/sample_12.wav"
SAMPLE_CLEAN = "./voicebank_wav/clean/sample_12.wav"
N_FFT = 512
HOP_LENGTH = 256

def get_stft(waveform):
    waveform = torch.from_numpy(waveform).float()
    if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
    
    max_amp = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / max_amp
    
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                      window=torch.hann_window(N_FFT), return_complex=True, normalized=True)
    return stft, max_amp

def analyze():
    # 1. Load Model
    device = torch.device('cpu')
    model = DeepComplexUNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # 2. Load Audio
    noisy_np, _ = sf.read(SAMPLE_NOISY)
    clean_np, _ = sf.read(SAMPLE_CLEAN)
    # 3. Calculate STFTs
    noisy_stft, _ = get_stft(noisy_np)
    clean_stft, _ = get_stft(clean_np)
    # 4. Calculate ideal CRM (Ground Truth)
    # Formula: Clean / Noisy
    eps = 1e-8
    ideal_crm = clean_stft / (noisy_stft + eps)
    # Clip to -1, 1 as the model is bounded by Tanh (BDT)
    ideal_crm_real = torch.clamp(ideal_crm.real, -1, 1).numpy().flatten()
    ideal_crm_imag = torch.clamp(ideal_crm.imag, -1, 1).numpy().flatten()
    
    # 5. Get MODEL CRM (Prediction)
    with torch.no_grad():
        # Normalise STFT as per model training
        norm_factor = torch.max(torch.abs(noisy_stft)) + eps
        real_in = (noisy_stft.real / norm_factor).unsqueeze(1)
        imag_in = (noisy_stft.imag / norm_factor).unsqueeze(1)
        
        mask_real, mask_imag = model(real_in, imag_in)
        pred_real = mask_real.squeeze().numpy().flatten()
        pred_imag = mask_imag.squeeze().numpy().flatten()
        
    # 6. PLOT SCATTER
    num_points = 5000 # Enough to see the trend, but safe for matplotlib
    idx = np.random.choice(len(ideal_crm_real), num_points, replace=False)
    plt.figure(figsize=(12, 5))
    # Real Part Correlation
    plt.subplot(1, 2, 1)
    plt.scatter(ideal_crm_real[idx], pred_real[idx], alpha=0.3, s=5, color='blue')
    plt.plot([-1, 1], [-1, 1], 'r--', label='Perfect (Identity)')
    plt.title("CRM Real Part: Industry vs. Model")
    plt.xlabel("Ideal CRM (Ground Truth)")
    plt.ylabel("U-Net Predicted CRM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Imaginary Part Correlation
    plt.subplot(1, 2, 2)
    plt.scatter(ideal_crm_imag[idx], pred_imag[idx], alpha=0.3, s=5, color='green')
    plt.plot([-1, 1], [-1, 1], 'r--', label='Perfect (Identity)')
    plt.title("CRM Imaginary Part (Phase): Industry vs. Model")
    plt.xlabel("Ideal CRM (Ground Truth)")
    plt.ylabel("U-Net Predicted CRM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle(f"Phase Consistency Analysis: {os.path.basename(SAMPLE_NOISY)}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_name = "phase_correlation_analysis.png"
    plt.savefig(plot_name, dpi=300)
    print(f"Analysis saved as: {plot_name}")
    plt.show()

    plt.figure(figsize=(10, 5))
    
    # Ground Truth Cloud
    plt.subplot(1, 2, 1)
    plt.scatter(ideal_crm_real[idx], ideal_crm_imag[idx], alpha=0.3, s=2, color='gray')
    plt.axhline(0, color='black', alpha=0.2)
    plt.axvline(0, color='black', alpha=0.2)
    plt.title("Ideal Mask Distribution\n(Ground Truth)")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)
    
    # Model Cloud
    plt.subplot(1, 2, 2)
    plt.scatter(pred_real[idx], pred_imag[idx], alpha=0.3, s=2, color='purple')
    plt.axhline(0, color='black', alpha=0.2)
    plt.axvline(0, color='black', alpha=0.2)
    plt.title("Model Mask Distribution\n(Your U-Net)")
    plt.xlabel("Real Part")
    plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    cloud_name = "mask_distribution_cloud.png"
    plt.savefig(cloud_name, dpi=300)
    plt.show()

if __name__ == "__main__":
    analyze()
