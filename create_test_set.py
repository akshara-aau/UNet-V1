import os
import random
import torchaudio
import torch
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
# Note: These paths should match where you stored your 83GB DNS dataset on the server
CLEAN_DIR = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/datasets_fullband/clean_fullband"
NOISE_DIR = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/datasets_fullband/noise_fullband"
TEST_SET_DIR = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/custom_test_set"

NUM_SAMPLES = 50 # How many test files you want

def create_test_set():
    print(f"🏗️  Initializing Test Set Generator...")
    os.makedirs(os.path.join(TEST_SET_DIR, "clean"), exist_ok=True)
    os.makedirs(os.path.join(TEST_SET_DIR, "noisy"), exist_ok=True)
    # 🕵️ Recursive Search for ALL .wav files (since DNS is nested)
    print(f"🔍 Searching for clean files in {CLEAN_DIR}...")
    clean_files = []
    for root, _, files in os.walk(CLEAN_DIR):
        for f in files:
            if f.endswith(".wav"):
                clean_files.append(os.path.join(root, f))
    
    print(f"🔍 Searching for noise files in {NOISE_DIR}...")
    noise_files = []
    for root, _, files in os.walk(NOISE_DIR):
        for f in files:
            if f.endswith(".wav"):
                noise_files.append(os.path.join(root, f))
    
    if not clean_files or not noise_files:
        print(f"❌ Error: Found {len(clean_files)} clean and {len(noise_files)} noise files. Check paths!")
        return

    print(f"🚀 Found {len(clean_files)} speech and {len(noise_files)} noise files.")
    print(f"🏗️  Creating {NUM_SAMPLES} test samples at 0dB SNR...")
    
    for i in tqdm(range(NUM_SAMPLES)):
        # 1. Pick a unique random sample for each test file
        clean_path = random.choice(clean_files)
        noise_path = random.choice(noise_files)
        
        # 2. Load audio using soundfile (Native bypass of torchaudio bugs)
        clean_np, sr = sf.read(clean_path)
        noise_np, _ = sf.read(noise_path)
        
        # Convert numpy to Torch (1, Channels, Samples)
        clean_wav = torch.from_numpy(clean_np).float()
        noise_wav = torch.from_numpy(noise_np).float()
        
        # Ensure it is (Channel, Time)
        if clean_wav.ndim == 1: clean_wav = clean_wav.unsqueeze(0)
        else: clean_wav = clean_wav.transpose(0, 1)
        
        if noise_wav.ndim == 1: noise_wav = noise_wav.unsqueeze(0)
        else: noise_wav = noise_wav.transpose(0, 1)
        # 3. Match lengths (5 seconds max for faster evaluation)
        min_len = min(clean_wav.shape[1], noise_wav.shape[1], 16000*5) 
        clean_wav = clean_wav[:, :min_len]
        noise_wav = noise_wav[:, :min_len]
        
        # 4. Mix at 0dB (Equal volume for a rigorous test)
        clean_rms = clean_wav.pow(2).mean().sqrt()
        noise_rms = noise_wav.pow(2).mean().sqrt()
        snr_factor = clean_rms / (noise_rms + 1e-8)
        
        # Simple mixing: Noisy = Clean + Noise (normalized to clean volume)
        noisy_wav = clean_wav + (noise_wav * snr_factor)
        
        # 5. Save using soundfile (Native bypass of torchaudio.save)
        # Note: sf.write expects (Samples, Channels) and numpy
        clean_out = clean_wav.transpose(0, 1).cpu().numpy()
        noisy_out = noisy_wav.transpose(0, 1).cpu().numpy()
        
        sf.write(os.path.join(TEST_SET_DIR, "clean", f"sample_{i}.wav"), clean_out, sr)
        sf.write(os.path.join(TEST_SET_DIR, "noisy", f"sample_{i}.wav"), noisy_out, sr)

    print(f"\n✅ SUCCESS: Custom test set created at: {TEST_SET_DIR}")
    print("👉 Now you can run: python3 complex_eval.py")

if __name__ == "__main__":
    create_test_set()
