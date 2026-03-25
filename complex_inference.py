import torch
import torchaudio
from complex_model import DeepComplexUNet

def infer_complex_audio(noisy_wav_path, model_path, output_path="complex_cleaned_result.wav"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load DCUNet Model
    model = DeepComplexUNet(n_channels=1)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}.")

    # 2. Audio settings
    sample_rate = 16000
    n_fft = 512
    hop_length = 256
    
    # Load and prep audio
    noisy_waveform, sr = torchaudio.load(noisy_wav_path)
    if sr != sample_rate:
        noisy_waveform = torchaudio.transforms.Resample(sr, sample_rate)(noisy_waveform)
    
    if noisy_waveform.shape[0] > 1:
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

    # 3. Apply STFT 
    stft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, 
        hop_length=hop_length, 
        power=None, 
        normalized=True # Must match complex_data_prep!
    )
    
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft, 
        hop_length=hop_length, 
        normalized=True
    )

    complex_spectrogram = stft(noisy_waveform)
    
    eps = 1e-8
    normalize_factor = torch.max(torch.abs(complex_spectrogram)) + eps
    
    # Isolate Real and Imaginary and Normalize
    mix_real = complex_spectrogram.real / normalize_factor
    mix_imag = complex_spectrogram.imag / normalize_factor
    
    # Add batch & channel dims [Batch, Channels, Freq, Time]
    mix_real_input = mix_real.unsqueeze(0).to(device)
    mix_imag_input = mix_imag.unsqueeze(0).to(device)
    
    # 4. Predict
    with torch.no_grad():
        pred_clean_real, pred_clean_imag = model(mix_real_input, mix_imag_input)
    
    # Denormalize
    pred_clean_real = pred_clean_real.squeeze(0) * normalize_factor
    pred_clean_imag = pred_clean_imag.squeeze(0) * normalize_factor
    
    # 5. Re-combine into a raw Complex Tensor
    # We no longer need Griffin-Lim because the model PREDICTED the phase geometry for us!
    cleaned_complex = torch.complex(pred_clean_real, pred_clean_imag).cpu()
    
    # 6. Apply Inverse STFT
    cleaned_waveform = istft(cleaned_complex)

    print(f"Saving DCUNet cleaned audio to {output_path}")
    torchaudio.save(output_path, cleaned_waveform, sample_rate)

if __name__ == "__main__":
    # Test 1
    infer_complex_audio(
       noisy_wav_path="noise_mixi.wav", 
       model_path="./complex_checkpoints/dcunet_epoch_100.pth", 
       output_path="cleaned_noise_mixi.wav"
    )

    # Test 2
    infer_complex_audio(
       noisy_wav_path="noise_baby.wav", 
       model_path="./complex_checkpoints/dcunet_epoch_100.pth", 
       output_path="cleaned_noise_baby.wav"
    )

    # Test 3
    infer_complex_audio(
       noisy_wav_path="wind_noise1.wav", 
       model_path="./complex_checkpoints/dcunet_epoch_100.pth", 
       output_path="cleaned_wind_noise1.wav"
    )

    # Test 4
    infer_complex_audio(
       noisy_wav_path="wind_noise2.wav", 
       model_path="./complex_checkpoints/dcunet_epoch_100.pth", 
       output_path="cleaned_wind_noise2.wav"
    )

    # Test 5
    infer_complex_audio(
       noisy_wav_path="wind_noise3.wav", 
       model_path="./complex_checkpoints/dcunet_epoch_100.pth", 
       output_path="cleaned_wind_noise3.wav"
    )
