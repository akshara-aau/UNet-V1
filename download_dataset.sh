#!/bin/bash
# Script to download the DNS-Challenge dataset on the AAU GPU cluster

# Make sure we fail fast if a command errors out
set -e

# Define where you want the dataset to live on the server
# You can change this path when you run it, e.g., using $WORKDIR on the cluster
DATASET_DIR=${1:-"./DNS-Challenge"}

echo "Starting setup for Microsoft DNS-Challenge Dataset..."
echo "Target directory: $DATASET_DIR"

if [ -d "$DATASET_DIR" ]; then
    echo "Directory $DATASET_DIR already exists. Skipping clone."
else
    # Clone the dataset directly from Microsoft's official GitHub repository
    echo "Cloning repository..."
    git clone https://github.com/microsoft/DNS-Challenge.git "$DATASET_DIR"
fi

cd "$DATASET_DIR"

echo "Note: The DNS-Challenge dataset is massive (hundreds of GBs)."
echo "You will need to manually run the specific python download scripts"
echo "provided in the repository (e.g., python download-dns-challenge-4.py)."
echo "After downloading the required tracks, you will run noisyspeech_synthesizer.py"
echo "to generate the paired noisy and clean speech datasets."
echo "Your data will typically be generated in:"
echo "  $DATASET_DIR/datasets/clean/"
echo "  $DATASET_DIR/datasets/noisy/"
echo "Make sure to update complex_train.py paths to point to these!"
