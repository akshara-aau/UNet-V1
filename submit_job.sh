#!/bin/bash
#SBATCH --job-name=unet_mtl_speech    # Job name
#SBATCH --nodes=1                     # Run all processes on a single node	
#SBATCH --ntasks=1                    # Run a single task		
#SBATCH --cpus-per-task=4             # Number of CPU cores per task (matches num_workers in data_prep.py)
#SBATCH --gres=gpu:1                  # Request 1 GPU. E.g., gpu:rtx3090:1 if you need a specific type
#SBATCH --mem=32G                     # Job Memory limit
#SBATCH --time=12:00:00               # Time limit hrs:min:sec (e.g., 12 hours max)
#SBATCH --output=unet_training_%j.out # Standard output and error log (%j is replaced with job ID)

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"

# 1. Load any necessary system modules (Python, CUDA, etc.)
# Uncomment and adjust these lines based on your specific AAU lab cluster requirements:
# module load python/3.10
# module load cuda/11.8

# 2. Define the project directory
# Adjust this to where you uploaded the files on the server!
WORKDIR="/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet2/mini-project-unet3"
cd $WORKDIR

# 3. Activate Virtual Environment from the parent directory
# Since we created a new nested folder (unet3), we can just re-use the python
# environment you already built in the old unet2 folder!
if [ -f "../unet_env/bin/activate" ]; then
    echo "Activating virtual environment from parent unet2 folder..."
    source ../unet_env/bin/activate
else
    echo "WARNING: Virtual environment 'unet_env' not found. Ensure required modules are installed."
fi

# 4. Link the Training Dataset
# This creates a symbolic link to the MS-SNSD dataset located one directory up!
if [ ! -d "./MS-SNSD" ]; then
    echo "Creating a symbolic link to the dataset in the parent folder..."
    ln -s ../MS-SNSD ./MS-SNSD
fi

# 5. Run the training script
echo "Starting multi-task U-Net training..."
python3 complex_train.py

echo "Job completed on $(date)"
