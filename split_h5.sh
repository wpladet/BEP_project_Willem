#!/bin/bash

#SBATCH --job-name=split_features
#SBATCH --output=job_logs/split_features%j.txt
#SBATCH --partition=tue.default.q
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G


source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh

# Import CUDA
module load CUDA/11.7.0

# Activate conda environment
conda activate thesis-conda-env6

# Check python version
which python

# Open WSI-finetuning project
cd /home/mcs001/20181133/CLAM/

# export CUDA_VISIBLE_DEVICES=0, 1
mv '/home/mcs001/20181133/CLAM/data_feat/Camelyon16_patch256_ostu_res50/h5_files' '/home/mcs001/20181133/CLAM/data_feat/Camelyon16_patch256_ostu_res50/h5_files_original'
python split_h5.py
