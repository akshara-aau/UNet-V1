#!/bin/bash
#SBATCH --job-name=boost_1_audioset
#SBATCH --output=boost_1_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=l4

WORKDIR="/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4"
cd $WORKDIR
OUTPUT_PATH="./datasets_fullband"
AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

# 🛰️ Focus: The massive Audioset 000 (Current Bottleneck)
BLOB="noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2"

echo "🛰️ Job 1: Starting Audioset 000 Boost [$(date)]"
if [ -f "$OUTPUT_PATH/${BLOB}.done" ]; then
    echo "✅ Already completed"
else
    curl -Lf "$AZURE_URL/$BLOB" | tar -C "$OUTPUT_PATH" --strip-components=1 -f - -x -j && touch "$OUTPUT_PATH/${BLOB}.done"
fi
