```

# File structure

LightGlu3D/
├── __pycache__
├──── extract_query_set.cpython-310.pyc
├──── generate_gt_pairs.cpython-310.pyc
├──── triangular_hloc.cpython-310.pyc
├──── visual_sfm_3d.cpython-310.pyc
├── split_query_ref
├──── extract_query_sets.py         # Split images into queries and references
├── triangulation
├──── run_sfm_cpu.sh                # Shell script to run sbatch CPU job
├──── run_sfm_gpu.sh                # Shell script to run sbatch GPU job
├──── triangulation_cpu_steps.py    # Triangulate and visualize SfM models (CPU)
├──── triangulation_gpu_steps.py    # Feature extraction & matching (GPU)
├──── utils.py                      # Utility functions
├──── view_3d_rotate.py
├──── visual_sfm_3d.py              # The visualization ply and html files
├── covisibility
├──── covisibility_search_pipe.py   # Filter block for query-relevant 3D points and references
├──── check_covisibility_thres.py
├── feature
├──── precompute_features.py        # Cache the averaged descriptors for 3D points
├── ground_truth
├──── generate_gt_pairs.py          # Base function to generate Ground Truth pairs, further applied as dataloader in gluefactory
├──── generate_gt_pairs_by_scene.py # Faster
├── jupyter_pipeline
├──── 2d_3d_matching_test.ipynb
├──── run_2d3d_matching_visual.ipynb
├──── run_sfm_visualization.ipynb   # Notebook for SfM visualization (pre/post search)
├── baseline
├──── feature_3d_compute.py         # Old experiment on projecting 3D to an image, for baseline use
├──── image_retrival.py             # Old experiment on projecting 3D to an image, for baseline use
├──── preprocessing.ipynb           # Old experiment on projecting 3D to an image, for baseline use
├──── rr_baseline.py                # RR (Rotate + Remove one coordinator)
├──── pr_baseline.py                # PR (Projection to Reference pose) 
├──── rn_baseline.py                # RN (Rotate + Normalization)
├── visualization
├──── network_weights               # The .tar file
├──── rerun_tools.py                # Help rurun file
├──── rerun_johanna.py              # Help rurun file
├──── visualize_normalization.py    # Visualize the normalization of Lightglu3d
├──── visualize_matches.py          # Visualize the predicted matches from baselines or trained model


# Merge with Jia in Github

# Berzelius cluster login
password: lsy20020409
code: (from NSC Berzelius:x_lishu)

cd matching/colla_preprocess/3d-2d-Matching-
cd matching/colla_gluefactory/glue-factory-2d3d-match

---

# Git operations for preprocess
# git init
# git remote add origin https://github.com/Jia0603/3d-2d-Matching-.git
# git fetch origin
# git checkout -b lsy-merged origin/lsy-merged
git checkout lsy-merged
git config --global user.name "Shuying Liu"
git config --global user.email "liushuying.blaise.2490@gmail.com"
git add .
git commit -m "pose estimation and changed hloc pipeline"
git push origin lsy-merged --force

# Git operations for training
# git init
# git clone https://github.com/Jia0603/glue-factory-2d3d-match.git
git pull
git config --global user.name "Shuying Liu"
git config --global user.email "liushuying.blaise.2490@gmail.com"
git add .
git commit -m ""
git push


---

# Get GPU
interactive --gpus=1 -t 4:00:00
# Get CPU (for triangulation)
interactive -p berzelius-cpu -t 8:00:00
# Check my use
squeue -u x_lishu
# cancel resource
scancel 15915922
# Check disk storage
nscquota

---

# Environment (install process in lsy-old)
mamba activate matchenv

---

# Split query images (split_query_ref)
# Change the way to get scene, from scene list or single scene for test
# Change to accpet arguments for ratios
# Change logger and tqdm

# Single scene
# Valid
python -m split_query_ref.extract_query_sets_re \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0022

# Test
python -m split_query_ref.extract_query_sets_re \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0025

# Train scene list
python -m split_query_ref.extract_query_sets_re \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

# Output
ls /proj/vlarsson/users/x_lishu/colla_matching/outputs/query

---

# Triangulation

# GPU steps (feature extraction and matching)
python -m triangulation.triangulation_gpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0022
python -m triangulation.triangulation_gpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

# CPU steps (triangulation)
python -m triangulation.triangulation_cpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --html_save_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/saved_html_visual \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

python -m triangulation.triangulation_cpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --html_save_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/saved_html_visual \
 --scene 0022

# Output
ls /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation

---

# Covisibility

covisibility_search_pipe_re.py

python -m covisibility.covisibility_search_pipe_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_list  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

python -m covisibility.covisibility_search_pipe_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_list  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0022

# Covisibility process for Aachen dataset
python -m covisibility.covisibility_search_pipe_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --outputs /proj/vlarsson/outputs_aachen/covisibility \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --query list /proj/vlarsson/datasets/aachen_v1.1/queries

---

# Feature computation

python -m feature.precompute_features_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

# Feature computation for Aachen
python -m feature.precompute_features_aachen

---

# Ground Truth

python -m ground_truth.generate_gt_pairs_re \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --feature_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

python -m ground_truth.generate_gt_pairs_re \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --feature_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --scene 0022

# Soft threshold
# Defult 
python -m ground_truth.generate_gt_pairs_soft \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --feature_dir /proj/vlarsson/outputs/midterm_results \
 --scene 0036

# output
ls /proj/vlarsson/users/x_lishu/colla_matching/outputs/feature

---

# Train

# Train with a adaptor MLP
# Only train on adaptor MLP, position, Last two layers of Lightglue
# v1: only adaptor and position
# v2: adaptor, position and two last layers, lr = 1e-4
python -m gluefactory.train lightglue_adapt_v2     --conf gluefactory/configs/2d_3d_lightglue_SP_finetune.yaml
# v3: adaptor, position and two last layers, soft threshould on gt, lr = 1e-5
python -m gluefactory.train lightglue_adapt_v3     --conf gluefactory/configs/2d_3d_lightglue_adapt_SP_finetune.yaml

# Tensorboard
tensorboard --logdir ~/matching/colla_gluefactory/glue-factory-2d3d-match/outputs/training/lightglue_adapt_v3 --port 6008

---

# NN baseline grid search
python -m gluefactory.run_nn_baseline_grid

# Change the path inside
python -m gluefactory.run_nn_baseline

# RR baseline
python -m baseline.rr_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir  /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt

# PR baseline
python -m baseline.pr_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt

# RN baselin
python -m baseline.rn_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir  /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt

---

# Visualization

# Visulaization of normalization
python -m visualization.visualize_normalization \
  --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
  --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
  --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
  --scene 0022

# Visulization of matches
python -m visualization.visualize_matches \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene 0022 \
 --method {change the method here}

python -m visualization.visualize_matches \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene 0022 \
 --method ADAPT \
 --checkpoint /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/visualization/network_weights/checkpoint_best_adapt.tar

# Visualize training
# Chnage this to add the custom path before visualizing TRAIN
export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"
# Put the .tar file into visulization.network_weights folder
python -m visualization.visualize_matches \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene 0022 \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/visualization/network_weights/checkpoint_best_clean.tar

# Visualize ground truth = 0
# Before becasue added the extra line so it will collapse
python -m visualization.visualize_no_gt \
  --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
  --covisibility_dir /proj/vlarsson/outputs/midterm_results \
  --query_dir  /proj/vlarsson/outputs/query_sets \
  --sfm_dir  /proj/vlarsson/outputs/sfm \
  --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
  --scene 0022

# Visualize abnormal superpoint keypoints
python -m visualization.visualize_2d_keypoint \
  --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
  --query_dir /proj/vlarsson/outputs/query_sets \
  --scene 0022

# Visualize ground truth with soft threshold
python -m visualization.visualize_gt \
  --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
  --covisibility_dir /proj/vlarsson/outputs/midterm_results \
  --query_dir  /proj/vlarsson/outputs/query_sets \
  --sfm_dir  /proj/vlarsson/outputs/sfm \
  --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
  --scene 0022

# Visualization of pose estimation
python -m visualization.visualize_pose_estimation \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --scene 0022 \
 --method NN \
 --max_error 12

---

# Estimation
# Change this to add the custom path before visualizing TRAIN
export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

python -m evaluation.pose_estimation \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --scene_list /proj/vlarsson/outputs/splits/val.txt \
 --method RR \
 --max_error 12 

python -m evaluation.pose_estimation \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --scene_list /proj/vlarsson/outputs/splits/val.txt \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/visualization/network_weights/checkpoint_best.tar

python -m evaluation.pose_estimation \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --scene_list /proj/vlarsson/outputs/splits/val.txt \
 --method HLOC \
 --max_error 6 

# Use Aachen for proper pose estimation, because magadepth dataset has unsure unit for transition




```