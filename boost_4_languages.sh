#!/bin/bash
#SBATCH --job-name=boost_4_languages
#SBATCH --output=boost_4_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=l4

WORKDIR="/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4"
cd $WORKDIR
OUTPUT_PATH="./datasets_fullband"
AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

# 🛰️ Focus: Extra Languages and Dev Test Set
BLOB_NAMES=(
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_000_NA_NA.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_001_NA_NA.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_000_0.00_3.47.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_001_3.47_3.64.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_000_0.00_3.98.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_000_0.00_4.09.tar.bz2"
    "datasets_fullband.dev_testset_000.tar.bz2"
)

echo "🛰️ Job 4: Starting Multi-Language Boost [$(date)]"

for BLOB in "${BLOB_NAMES[@]}"
do
    echo "⬇️ [$(date '+%H:%M:%S')] Processing: $BLOB"
    if [ -f "$OUTPUT_PATH/${BLOB}.done" ]; then
        echo "✅ Already completed"
        continue
    fi
    curl -Lf "$AZURE_URL/$BLOB" | tar -C "$OUTPUT_PATH" --strip-components=1 -f - -x -j && touch "$OUTPUT_PATH/${BLOB}.done"
done
