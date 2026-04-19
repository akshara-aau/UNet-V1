#!/bin/bash
#SBATCH --job-name=boost_2_noise
#SBATCH --output=boost_2_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=l4

WORKDIR="/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4"
cd $WORKDIR
OUTPUT_PATH="./datasets_fullband"
AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

#  Focus: The remaining noise blobs
BLOB_NAMES=(
    "noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2"
    "noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2"
    "noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2"
    "noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2"
)

echo " Job 2: Starting Noise Boost (Audioset 001-002, Freesound) [$(date)]"

for BLOB in "${BLOB_NAMES[@]}"
do
    echo " [$(date '+%H:%M:%S')] Processing: $BLOB"
    if [ -f "$OUTPUT_PATH/${BLOB}.done" ]; then
        echo " Already completed"
        continue
    fi
    curl -Lf "$AZURE_URL/$BLOB" | tar -C "$OUTPUT_PATH" --strip-components=1 -f - -x -j && touch "$OUTPUT_PATH/${BLOB}.done"
done
