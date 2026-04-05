#!/bin/bash
#SBATCH --job-name=boost_3_ir_speech
#SBATCH --output=boost_3_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=l4

WORKDIR="/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4"
cd $WORKDIR
OUTPUT_PATH="./datasets_fullband"
AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

# 🛰️ Focus: Impulse Responses + English Speech
BLOB_NAMES=(
    "datasets_fullband.impulse_responses_000.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_000.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_001.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2"
)

echo "🛰️ Job 3: Starting IR + English Speech Boost [$(date)]"

for BLOB in "${BLOB_NAMES[@]}"
do
    echo "⬇️ [$(date '+%H:%M:%S')] Processing: $BLOB"
    if [ -f "$OUTPUT_PATH/${BLOB}.done" ]; then
        echo "✅ Already completed"
        continue
    fi
    curl -Lf "$AZURE_URL/$BLOB" | tar -C "$OUTPUT_PATH" --strip-components=1 -f - -x -j && touch "$OUTPUT_PATH/${BLOB}.done"
done
