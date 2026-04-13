import pandas as pd
import soundfile as sf
import io
import os
from tqdm import tqdm

# Configuration
PARQUET_FILE = './voicebank_demand_16k/data/test-00000-of-00001.parquet'
OUTPUT_BASE = './voicebank_wav/test'

def extract():
    print("📦 Starting Direct Parquet Extraction...")
    os.makedirs(f"{OUTPUT_BASE}/noisy", exist_ok=True)
    os.makedirs(f"{OUTPUT_BASE}/clean", exist_ok=True)

    if not os.path.exists(PARQUET_FILE):
        print(f"❌ Error: File {PARQUET_FILE} not found. Check your path!")
        return

    df = pd.read_parquet(PARQUET_FILE)
    print(f"🚀 Found {len(df)} samples. Extracting now...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Extract binary data
        noisy_bytes = row['noisy']['bytes']
        clean_bytes = row['clean']['bytes']
        
        # Decode using soundfile
        noisy_data, sr = sf.read(io.BytesIO(noisy_bytes))
        clean_data, _ = sf.read(io.BytesIO(clean_bytes))
        
        # Save as physical .wav files 
        sf.write(f'{OUTPUT_BASE}/noisy/sample_{i}.wav', noisy_data, sr)
        sf.write(f'{OUTPUT_BASE}/clean/sample_{i}.wav', clean_data, sr)

    print(f"✅ Success! Your test set is in {OUTPUT_BASE}")

if __name__ == "__main__":
    extract()
