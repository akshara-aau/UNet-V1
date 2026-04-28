#!/bin/bash
#SBATCH --job-name=wind_specialist
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --output=wind_transfer_%j.log

# FORCE STABLE AUDIO BACKENDS
export TORCHAUDIO_USE_BACKEND_DISPATCHER=1
export TORCH_AUDIO_BACKEND="sox_io"

echo "Starting Wind Transfer Learning..."
echo "Job ID: $SLURM_JOB_ID"

# 1. Activate Environment
# Using the absolute path to your environment on the AAU cluster
source /ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/unet_env/bin/activate

# 2. Run Training
# Make sure complex_train.py is in the same folder as this script
python3 complex_train.py

echo "Training Job Complete."
