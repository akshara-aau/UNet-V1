# Deep Complex U-Net for Wind Noise Suppression

This repository contains Deep Complex U-Net for general noise suppression.

Project Overview

Deep Lerning model based on DCUnet architecture for general noise suppression.
 The model uses:
 - Complex Ratio Masking (CRM)
 - wSDR loss

Project Structure 

1. complex_model.py: 
Core model architecture with:

Complex Convolutions
Complex Batch Normalization
Encoder-Decoder Skip Connections

2. complex_data_prep.py: 

Audio preprocessing pipeline:

Mono conversion
16 kHz resampling
Peak normalization
4-second segmentation

3. complex_train.py: 
Training pipeline using:

Complex spectrogram processing and wSDR loss optimization

4. complex_local_inference.py: 

Inference script for enhancing external .wav files using trained model weights.

5. plot_loss.py: 

Plots training and validation loss curves from training logs.

1.Install Dependencies

pip install -r requirements.txt


How to Train

If you want to train the model from scratch:

1. Prepare Your Dataset:
   Organize your data into specific folders (Clean speech, Noise, and RIRs).
   
2. Configure Paths
   Open complex_train.py and update the `CLEAN_DIR`, `NOISE_DIR`, and `RIR_DIR` variables.

3. Start Training:
   sbatch submit_job.sh
   Checkpoints and logs will be saved to `complex_checkpoints/`.

How to Test

1. Place the trained model checkpoint (for example, dcunet_epoch_120.pth) inside:
complex_checkpoints/

2. Open complex_local_inference.py and update:

noisy_wav_path
output_path

3. Run inference:
python3 complex_local_inference.py



