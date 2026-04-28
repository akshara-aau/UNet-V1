import os
import tqdm
import soundfile as sf
from rwth_wind_generator import RWTHWindGenerator

def main():
    # Configuration
    OUTPUT_DIR = "synthetic_wind_dataset"
    NUM_FILES = 2500  # Total 10-second files
    DURATION = 10     #10 s files; model will take snaps from different part of the clip if its lengthy
    FS = 16000
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"start generating wind dataset")
    gen = RWTHWindGenerator(model_path='wind_noise_model')
    
    print(f"Generating {NUM_FILES} files of {DURATION}s each")
    # Loop with a progress bar
    for i in tqdm.tqdm(range(NUM_FILES)):
        filename = f"wind_synth_{i:04d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if already exists (resumeable)
        if os.path.exists(filepath):
            continue
            
        try:
            # Generate the natural wind audio
            audio = gen.generate(DURATION, FS)
            
            # Save using soundfile
            sf.write(filepath, audio, FS)
        except Exception as e:
            print(f"Error generating file {i}: {e}")
            continue

    print(f"\nSUCCESS: Dataset generated in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
