LightGlu3D
|_____extract_query_sets.py # First split the images into queries and references
|_____triangulation_gpu_steps.py # Image pair retrieval based on covisibility, then do feature extraction, matching to get keypoint matches
|_____triangulation_cpu_steps.py # Triangulate and visualize SfM models
|_____precompute_features.py # Cache the averaged descriptor for 3D points
|_____covisibility_searh_pipe.py # Filter block where we do covisibility search to get query-relevant 3D points and references
|_____run_sfm_visualization.ipynb # Python notebook for SfM visualization use (before and after covisibility search)
|_____add variance to the 3D descritors, will consider later on how to conduct(UNDO)

|_____utils.py # Tool fuctions
|_____run_sfm_gpu.sh # Script to run sbatch job
|_____run_sfm_cpu.sh # Script to run sbatch job
