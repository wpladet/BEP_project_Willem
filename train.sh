#!/bin/bash

#SBATCH --job-name=train_abmil_CAM16
#SBATCH --output=job_logs/train_abmil%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1

source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh

module load CUDA/11.7.0
conda activate thesis-conda-env6
which python

# Open WSI-finetuning project
cd /home/mcs001/20181133/CLAM/

export CUDA_VISIBLE_DEVICES=0

python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 5 --exp_code camelyon16_abmil_50 --weighted_sample --bag_loss ce --task task_1_tumor_vs_normal --model_type abmil --log_data --data_root_dir /home/mcs001/20181133/CLAM/data_feat --embed_dim 1024

