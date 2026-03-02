### File Structure

```text
LightGlu3D/
├── __pycache__
├──── extract_query_set.cpython-310.pyc
├──── generate_gt_pairs.cpython-310.pyc
├──── triangular_hloc.cpython-310.pyc
├──── visual_sfm_3d.cpython-310.pyc
├── split_query_ref
├──── extract_query_sets.py         # Split images into queries and references
├── covisibility
├──── covisibility_search_pipe.py   # Filter block for query-relevant 3D points and references
├──── check_covibility_thres.py
├── triangulation
├──── run_sfm_cpu.sh                # Shell script to run sbatch CPU job
├──── run_sfm_gpu.sh                # Shell script to run sbatch GPU job
├──── triangulation_cpu_steps.py    # Triangulate and visualize SfM models (CPU)
├──── triangulation_gpu_steps.py    # Feature extraction & matching (GPU)
├──── utils.py                      # Utility functions
├──── view_3d_rotate.py
├──── visual_sfm_3d.py
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


'''