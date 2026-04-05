#!/bin/bash
#SBATCH --job-name=dns_download
#SBATCH --output=download_dns_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=l4    

echo "Starting MS-DNS Download..."
# Navigate to your project root
cd /ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/

# The script confirms the repo exists and triggers download
if [ -d "DNS-Challenge" ]; then
    cd DNS-Challenge
    # Make sure the script is executable
    chmod +x download-dns-challenge-4.sh
    # Trigger the massive non-interactive pull
    bash download-dns-challenge-4.sh
else
    echo "ERROR: DNS-Challenge folder not found. Clone it manually first."
fi
