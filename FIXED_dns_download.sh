#!/bin/bash
#SBATCH --job-name=dns_download
#SBATCH --output=dns_download_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=l4

# --- 🏗️ TARGET PROJECT PATH ---
WORKDIR="/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4"
cd $WORKDIR

# --- 📂 OUTPUT SETUP ---
OUTPUT_PATH="./datasets_fullband"
mkdir -p "$OUTPUT_PATH"

AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

# --- 📜 VERIFIED MASTER LIST (DNS-4 ICASSP 2022) ---
BLOB_NAMES=(
    # 🕵️‍♂️ PRIORITY 1: NOISE (Required for Synthesis)
    "noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2"
    "noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2"
    "noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2"
    "noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2"
    "noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2"
    
    # 🕵️‍♂️ PRIORITY 2: IMPULSE RESPONSES
    "datasets_fullband.impulse_responses_000.tar.bz2"

    # 🕵️‍♂️ PRIORITY 3: SPEECH (Verified Suffixes)
    "clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_000.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_001.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2"
    
    # 🕵️‍♂️ EXTRA LANGUAGES
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_000_NA_NA.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_001_NA_NA.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_000_0.00_3.47.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_001_3.47_3.64.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_000_0.00_3.98.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2"
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_000_0.00_4.09.tar.bz2"
    "datasets_fullband.dev_testset_000.tar.bz2"
)

echo "--- 🛰️ Starting Unified Master DNS Download Flow [$(date)] ---"

for BLOB in "${BLOB_NAMES[@]}"
do
    URL="$AZURE_URL/$BLOB"
    echo "⬇️ [$(date '+%H:%M:%S')] Processing: $BLOB"

    # Skip if 'receipt' exists
    if [ -f "$OUTPUT_PATH/${BLOB}.done" ]; then
        echo "✅ [SKIP] Already completed: $BLOB"
        continue
    fi

    echo "📡 [$(date '+%H:%M:%S')] Streaming Download + Direct Extraction..."
    
    # --strip-components=1 ignores the internal "datasets_fullband" folder to fix nesting
    curl -Lf "$URL" | tar -C "$OUTPUT_PATH" --strip-components=1 -f - -x -j \
        && touch "$OUTPUT_PATH/${BLOB}.done"

    if [ $? -eq 0 ]; then
        echo "🎉 [DONE] Successfully finished: $BLOB"
    else
        echo "❌ [ERROR] Could not process $BLOB"
    fi
    echo "-----------------------------------"
done

echo "--- ✨ Sequence Completed at $(date) ---"
