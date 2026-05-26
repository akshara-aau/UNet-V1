import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

# --- CONFIG ---
DATA_DIR = "test_audio_DNS_120th_result/fail-test/original/2nd-test"
# OUTPUT_FILE = "test_audio_DNS_120th_result/fail-test/model_comparison_log.png"
OUTPUT_FILE = "test_audio_DNS_120th_result/fail-test/model_comparison_log_2.png"


# Specific filenames provided by user
FILES = {
    # "I. Noisy Input": "bikas_1.wav",
    # "II. General DNS Model Output": "enhanced_DNS_bikas.wav",
    # "III. Specialized Wind Model Output": "enhanced_wind_model_bikas.wav"

    "I. Noisy Input": "hard_1.wav",
    "II. General DNS Model Output": "enhanced_hard_1.wav",
    "III. Specialized Wind Model Output": "enhanced_Wind_model.wav"
}

def plot_model_comparison():
    plt.style.use('default')
    plt.figure(figsize=(15, 18), facecolor='white')
    
    for i, (title, filename) in enumerate(FILES.items()):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Error: {filename} not found in {DATA_DIR}")
            continue

        print(f"Analyzing {filename}...")

        y, sr = sf.read(path)
        if len(y.shape) > 1: y = y[:, 0]

        n_fft = 1024
        f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft//2)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        plt.subplot(3, 1, i+1)
        
        plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=-110, vmax=-30)
        
        plt.yscale('log')
        plt.ylim(20, 8000) 
        
        plt.title(f"{title}", fontsize=15, fontweight='bold', pad=15)
        plt.ylabel("Frequency (Hz, log scale)", fontsize=12)
        if i == 2: plt.xlabel("Time (s)", fontsize=12)
        
        plt.colorbar(label='Magnitude (dB)')
        
        plt.yticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 8000], 
                   ['20', '50', '100', '200', '500', '1k', '2k', '5k', '8k'])

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    print(f"\nSUCCESS: Comparison plot (Log Scale) saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_model_comparison()
