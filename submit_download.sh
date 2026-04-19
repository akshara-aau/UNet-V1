#!/bin/bash
#SBATCH --job-name=dns_test_download
#SBATCH --output=download_test_set.log
#SBATCH --error=download_test_set.err
#SBATCH --mem=4G
#SBATCH --time=04:00:00          # 4 hours is plenty

echo " [$(date)] Download Started"

# 1. Navigate to the evaluation folder
mkdir -p ~/P8-AVS-WNS/mini-project-unet4/evaluation_test_set
cd ~/P8-AVS-WNS/mini-project-unet4/evaluation_test_set

# 2. Copy the script
cp ~/P8-AVS-WNS/mini-project-unet4/DNS-Challenge/download-dns5-dev-testset.sh .

# 3. Run the download
echo " Running download-dns5-dev-testset.sh..."
bash download-dns5-dev-testset.sh

echo " [$(date)] Download and Cleanup Completed!"
