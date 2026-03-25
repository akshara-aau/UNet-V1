#!/bin/bash
# Script to download the MS-SNSD dataset on the AAU GPU cluster

# Make sure we fail fast if a command errors out
set -e

# Define where you want the dataset to live on the server
# You can change this path when you run it, e.g., using $WORKDIR on the cluster
DATASET_DIR=${1:-"./MS-SNSD"}

echo "Starting download of Microsoft Scalable Noisy Speech Dataset (MS-SNSD)..."
echo "Target directory: $DATASET_DIR"

if [ -d "$DATASET_DIR" ]; then
    echo "Directory $DATASET_DIR already exists. Skipping download to prevent overwrite."
    exit 0
fi

# Clone the dataset directly from Microsoft's official GitHub repository
echo "Cloning repository..."
git clone https://github.com/microsoft/MS-SNSD.git "$DATASET_DIR"

# Navigate into the dataset directory
cd "$DATASET_DIR"

# By default, MS-SNSD has a script `noisyspeech_synthesizer.py` that generates 
# pre-mixed audio if you want it, but we only need the raw clean and noise files. 
# They are located in:
# MS-SNSD/clean_train/
# MS-SNSD/noise_train/
# MS-SNSD/clean_test/
# MS-SNSD/noise_test/

echo "Download complete!"
echo "Your raw clean speech files are in: $DATASET_DIR/clean_train/"
echo "Your raw noise files are in: $DATASET_DIR/noise_train/"
echo "You can now point data_prep.py to these directories during training."
