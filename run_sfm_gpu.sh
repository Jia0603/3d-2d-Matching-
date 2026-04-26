#!/bin/bash
#SBATCH -A berzelius-2026-113 
#SBATCH -J sfm_hloc_gpu               
#SBATCH -t 00-01:00:00               
#SBATCH -o log_file/sfm_gpu_cambridge_4others%j.log

#SBATCH -p berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python sfm_gpu_cambridge.py
# python triangulation_gpu_steps.py