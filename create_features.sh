#!/bin/bash

#SBATCH --job-name=feature_extraction_CAM16
#SBATCH --output=job_logs/fe_job_output_resnet18_%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1

# Load necessary modules
source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh
module load CUDA/11.7.0

# Activate conda environment
conda activate BEP

# Check python version
which python

# Navigate to project directory
cd /home/bme001/20202047/CLAM/

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Run the python script in background
#python extract_features_fp.py --model_name resnet18_imagenet --data_h5_dir /home/bme001/20202047/patches/Camelyon16_patch256_ostu --data_slide_dir /home/tue/shared_data/ml-datasets/CAMELYON16/images --csv_path /home/bme001/20202047/patches/Camelyon16_patch256_ostu/process_list_autogen.csv --feat_dir /home/bme001/20202047/data_feat/Camelyon16_patch256_res18_imagenet --batch_size 256 --slide_ext .tif
python extract_features_fp.py --model_name resnet18 --data_h5_dir /home/bme001/20202047/patches/Camelyon16_patch256_ostu --data_slide_dir /home/tue/shared_data/ml-datasets/CAMELYON16/images --csv_path /home/bme001/20202047/patches/Camelyon16_patch256_ostu/process_list_autogen.csv --feat_dir /home/bme001/20202047/data_feat/Camelyon16_patch256_res18_ssl --batch_size 256 --slide_ext .tif
# nvidia-smi -a
# nvidia-smi mig -lgi  # List GPU instances

# Get the Python script's process ID
# PYTHON_PID=$!

# Monitor GPU usage every minute in the background
# while kill -0 $PYTHON_PID 2> /dev/null; do
#     nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 >> gpu_usage_$SLURM_JOB_ID.log
#     sleep 10
# done

# This loop will output the GPU utilization to a log file named 'gpu_usage_<jobID>.log'
# every minute until the Python script completes.
