#!/bin/bash

#SBATCH --job-name=segmenting_and_patching_CAM16
#SBATCH --output=job_logs/patching_job_output_%j.txt
#SBATCH --partition=tue.default.q
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G

# Load the Python module
# module load Python/3.9.5-GCCcore-10.3.0
# module load anaconda/2021.11-pth39
# python --version

# cd /home/bme001/20202047

# # Activate virtual environment
# conda init bash
# echo "bash init done"
# conda activate BEP
# which python

source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh
conda activate BEP
which python
# Open WSI-finetuning project
cd /home/bme001/20202047/CLAM/


# hopefull envirenment variables are now set and stuff can be located
# CUDA_VISIBLE_DEVICES=0 python create_patches_fp.py --source /home/tue/shared_data/ml-datasets/CAMELYON16/images --save_dir ./patches0/Camelyon16_patch256_ostu --patch_level 0 --patch_size 256 --step_size 256 --seg --patch --stitch --use_ostu 
CUDA_VISIBLE_DEVICES=0 python create_patches_fp.py --source /home/tue/shared_data/ml-datasets/CAMELYON16/images --save_dir /home/bme001/20202047/patches/Camelyon16_patch256_ostu --patch_size 256 --step_size 256 --preset bwh_biopsy.csv --seg --patch --stitch