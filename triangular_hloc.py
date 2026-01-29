from pathlib import Path
import torch
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from hloc import (
    pairs_from_retrieval,
    extract_features,
    match_features,
    triangulation,
    visualization,
)

def extract_pairs_to_list(npz_path, overlap_thres=[0.1, 0.5]):

    print(f"Using overlap threshold range: {overlap_thres}")

    full_data = np.load(npz_path, allow_pickle=True)
    overlap_matrix = full_data['overlap_matrix']
    valid_pairs = np.argwhere((overlap_matrix >= overlap_thres[0]) & (overlap_matrix < overlap_thres[1]))
    valid_pairs_path = full_data['image_paths'][valid_pairs]
    valid_pairs_name = [(os.path.basename(pair[0]), os.path.basename(pair[1])) for pair in valid_pairs_path]

    return valid_pairs_name


root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
outputs = Path("./outputs/sfm/")

overlap_thres = [0.1, 0.7]  # define your overlap threshold range here

feature_conf = {
    'output': 'feats-superpoint-n2048',
    'model': {
        'name': 'superpoint',
        'nms_radius': 4,
        'max_keypoints': 2048, 
    },
    'preprocessing': {
        'grayscale': True,
        'resize_max': 1600, 
        "resize_force": True,
    },
}

retrieval_conf = extract_features.confs["netvlad"]
matcher_conf = match_features.confs["superpoint+lightglue"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scene_names = sorted([
    p.name
    for p in root.iterdir()
    if p.is_dir()
])

for scene in scene_names[:1]: # change the slice to process more scenes
    print(f"Start processing scene: {scene}...")

    images_path = root / scene / "images"

    output_dir = outputs / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    if device.type == 'cuda':

        print("Using GPU for processing feature extraction and matching.")
        sfm_pairs = output_dir / "pairs-covisibility.txt"

        # # Step 0: Get image pairs from dataset info
        # retrieval_path = extract_features.main(retrieval_conf, images_path, output_dir)
        # pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=10)

        pair_load = extract_pairs_to_list(root.parent / 'scene_info' / f'{scene}.npz', overlap_thres)

        with sfm_pairs.open("w") as f:
            for im1, im2 in pair_load:
                f.write(f"{im1} {im2}\n")

        print(f"Finished similar pairs retrival. Processed {len(pair_load)} pairs.")

        # Step 1: Feature extraction
        feature_path = extract_features.main(feature_conf, images_path, output_dir)

        # Step 2: Pairwise matching
        match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], output_dir
        )

        print(f"Secene {scene} feature extraction and matching on GPU completed.")

    if device.type =='cpu':

        print("Using CPU for processing 3D triangulation.")
        sfm_pairs = output_dir / "pairs-covisibility.txt"
        sfm_dir = output_dir / "sfm_superpoint+lightglue"
        feature_path = output_dir / "feats-superpoint-n2048.h5"
        match_path = output_dir / "feats-superpoint-n2048_matches-superpoint-lightglue_pairs-covisibility.h5"
        reference_model = root / scene / "sparse"

        # Step 3: Triangulation to obtain 3D model
        model = triangulation.main(sfm_dir, reference_model, images_path, sfm_pairs, feature_path, match_path)
        
        print(f"Secene {scene} 3D triangulation on CPU completed.")

        # Step 4: Visualization
        
