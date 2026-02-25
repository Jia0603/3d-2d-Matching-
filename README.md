### File Structure

```text
LightGlu3D/
├── extract_query_sets.py        # Split images into queries and references
├── triangulation_gpu_steps.py   # Feature extraction & matching (GPU)
├── triangulation_cpu_steps.py   # Triangulate and visualize SfM models (CPU)
├── precompute_features.py       # Cache the averaged descriptors for 3D points
├── covisibility_search_pipe.py  # Filter block for query-relevant 3D points and references
├── run_sfm_visualization.ipynb  # Notebook for SfM visualization (pre/post search)
├── generate_gt_pairs.py         # Base function to generate Ground Truth pairs, further applied as dataloader in gluefactory
├── utils.py                     # Utility functions
├── run_sfm_gpu.sh               # Shell script to run sbatch GPU job
└── run_sfm_cpu.sh               # Shell script to run sbatch CPU job
├── feature_3d_compute.py        # Old experiment on projecting 3D to an image, for baseline use
├── image_retrival.py            # Old experiment on projecting 3D to an image, for baseline use
├── preprocessing.ipynb          # Old experiment on projecting 3D to an image, for baseline use
'''
