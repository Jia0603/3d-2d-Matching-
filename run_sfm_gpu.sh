#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J sfm_hloc_gpu               
#SBATCH -t 00-03:00:00               
#SBATCH -o sfm_output_%j.log

#SBATCH -p berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "Running in GPU mode on $HOSTNAME"

python triangular_hloc.py