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
├── feature_extract
├──── precompute_features.py        # Cache the averaged descriptors for 3D points
├── ground_truth
├──── generate_gt_pairs.py          # Base function to generate Ground Truth pairs, further applied as dataloader in gluefactory
├── jupyter_pipeline
├──── 2d_3d_matching_test.ipynb
├──── run_2d3d_matching_visual.ipynb
├──── run_sfm_visualization.ipynb   # Notebook for SfM visualization (pre/post search)
├── baseline
├──── feature_3d_compute.py         # Old experiment on projecting 3D to an image, for baseline use
├──── image_retrival.py             # Old experiment on projecting 3D to an image, for baseline use
├──── preprocessing.ipynb           # Old experiment on projecting 3D to an image, for baseline use
├──── getBaseline.py                # Calculate the baseline based on cosine similarity of descriptors
├── old                             # Not use, for reference only


# Merge with Jia in Github

# Berzelius cluster login
password: lsy20020409
code: (from NSC Berzelius:x_lishu)

cd matching
cd colla_preprocess/3d-2d-Matching-

cd colla_gluefactory/glue-factory-2d3d-match


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
git commit -m "New pull. Split query change."
git push origin lsy-merged --force

# Git operations for training

---

# Get GPU
interactive --gpus=1 -t 8:00:00
# Get CPU (for triangulation)
interactive -p berzelius-cpu -t 8:00:00
# Check my use
squeue -u x_lishu
# cancel resource
scancel 15672530

---

# Environment (install process in lsy-old)
mamba activate matchenv

# Preprocess

# Still use 10 scenes for training, 1 scene for validation, 1 for testing
# Change other scenes to try the pipeline
# Train list:
/home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt
# Valid list:
/home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/valid_scenes_clean_try.txt
# Test list:
/home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/test_scenes_clean_try.txt

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

---

# Feature computation


```