#!/bin/bash
#SBATCH --job-name=wind_factory
#SBATCH --output=wind_factory_%j.log
#SBATCH --error=wind_factory_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
source /ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/unet_env/bin/activate

echo "Starting Wind Dataset Generation Job"
python3 generate_wind_dataset.py
echo "Wind Dataset Generation Job Finished"
