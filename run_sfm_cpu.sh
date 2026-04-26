#!/bin/bash
#SBATCH -A berzelius-2026-113
#SBATCH -J sfm_hloc               
#SBATCH -t 00-03:00:00               
#SBATCH -o log_file/sfm_cpu_aachen_old%j.log

#SBATCH -p berzelius-cpu                    
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=4        
#SBATCH --mem=92G 
echo "Running in CPU mode on $HOSTNAME"

python aachen_retriangulation.py
# python sfm_cpu_cambridge.py
# python triangulation_cpu_steps.py