# Berzelius cluster
password: lsy20020409
code: (from NSC Berzelius:x_lishu)

# Install environment
# module load Mambaforge/23.3.1-1-hpc1-bdist # Add in .bashrc
# mamba create --name matchenv python=3.10
mamba activate matchenv
cd matching

---

# Git operations
# git init
# git remote add origin https://github.com/Jia0603/3d-2d-Matching-.git

---

# Environment
# LightGlue
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install -e .

# Gluefactory
cd ..
git clone https://github.com/cvg/glue-factory.git
cd glue-factory
python3 -m pip install -e .
# python3 -m pip install -e .[extra]

# Hloc
cd ..
git clone https://github.com/cvg/Hierarchical-Localization.git
cd Hierarchical-Localization
python -m pip install -e .
# The missing submodule
git submodule update --init --recursive

---

# Megadepth dataset
# Prepare the storage
cd /proj/vlarsson/users/x_lishu
mkdir -p data/megadepth
cd data/megadepth
# Start a new session named 'download'
tmux new -s download
# Download the SfM models
wget https://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz
# Request 1 CPU node from the cpu partition
interactive -p berzelius-cpu -t 04:00:00
# Extract .tar.xz files
tar -xvf MegaDepth_SfM_v1.tar.xz -C megadepth

# No need to download and extract, find in shared project storage
# Megadeoth path
ls -F /proj/vlarsson/datasets/megadepth/Undistorted_SfM/

---
# Get GPU
interactive --gpus=1 -t 2:00:00
# Get CPU (for triangulation)
interactive -p berzelius-cpu -t 12:00:00
# Check my use
squeue -u x_lishu
# cancel resource
scancel 15156639

mamba activate matchenv
cd matching

cd my_matching

# Triangulation
# Frist, find the match on GPU
python -m triangulation.findMatch \
    --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
    --outputs /proj/vlarsson/users/x_lishu/matching/outputs/triangulation \
    --scene 0008
    --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt
# Second, do the triangulation on CPU
python -m triangulation.triangulationSuperpoint \
    --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
    --outputs /proj/vlarsson/users/x_lishu/matching/outputs/triangulation \
    --scene 0015
    --scene_list  /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt
# Thrid, 
python -m feature.getFeature3D_tri \
    --outputs /proj/vlarsson/users/x_lishu/matching/outputs/triangulation \
    --scene 0003
# Get ground truth based on the new points3D.bin
python -m triangulation.getGroundTruth \
    --outputs /proj/vlarsson/users/x_lishu/matching/outputs/triangulation \
    --scene 0015
    --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt


mamba activate matchenv
cd matching
cd glue-factory
cd gluefactory

# Debug train

# Train
python -m gluefactory.train lightglue_3d_experiment_v1 \
    --conf configs/lightglu3d.yaml \
    --mixed_precision float16

# Tensorboard
tensorboard --logdir ~/matching/glue-factory/outputs/training/lightglue_3d_experiment_v1 --port 6006

# Check outputs in my shared storage
ls -F /proj/vlarsson/users/x_lishu/matching/outputs/triangulation

# Show the performance on test
python -m eval.evaluateTest

# Show some examples of trained network on test
python -m eval.visualize


---

# Baseline
python -m baseline.getBaseline \
    --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
    --outputs /proj/vlarsson/users/x_lishu/matching/outputs/triangulation \
    --scene 0008

---

# scene info
triangular_hloc.py

# visualize3d 

---

# Berzelius info
# The project info
projinfo 
# Disk sotrage
nscquota

---

# Ludde provided
lastjobs  # To see your last jobs
jobsh  # Login to compute node to check running job
squeue -u $USER  # info about the current jobs
scontrol show job JOBID  # info about a specific job
scancel JOBID  # cancel a job
# Request 1 GPU with defaults: 16 CPU cores, 128 GB RAM, and a default wall time of 2h.
interactive --gpus=1 -t 1-00:00:00
# Request 2 GPU for 30 minutes: 32 CPU cores, 256 GB RAM. The time limit format is days-hours:minutes:seconds.
interactive --gpus=2 -t 00-00:30:00
# Find out what to put in a .sh file
sbatch --help
# To start a job
sbatch jobs/glue_factory.sh
# Use several workers and divide up dataset into zip-files
fpsync -n 8 -m tarify -s 2350M -f 19000 /proj/vlarsson/datasets/hpatches-sequences-release/ /proj/vlarsson/datasets/hpatches-sequences-release_tars/
# In the batch job script, write e.g.
ls /proj/vlarsson/datasets/hpatches-sequences-release_tars/*.tar | xargs -n 1 -P 8 tar -x -C /scratch/local/ -f257710
# Log into node
jobsh -j $jobid
# Checking GPU usage on a node
nvidia-smi
nvtop
top  # to check CPU usage
htop  # this is a better version than top to check CPU usage. More user friendly.