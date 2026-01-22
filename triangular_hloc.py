from pathlib import Path
import torch

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    triangulation,
    visualization,
    pairs_from_retrieval,
)

root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
outputs = Path("outputs/sfm/")

retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superglue"]
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
        sfm_pairs = output_dir / "pairs-netvlad.txt"
        sfm_dir = output_dir / "sfm_superpoint+superglue" 
        # Step 0: Image retrieval to get image pairs
        retrieval_path = extract_features.main(retrieval_conf, images_path, output_dir)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

        # Step 1: Feature extraction
        feature_path = extract_features.main(feature_conf, images_path, output_dir)

        # Step 2: Pairwise matching
        match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], output_dir
        )

    if device.type =='cpu':

        print("Using CPU for processing 3D triangulation.")
        sfm_pairs = output_dir / "pairs-netvlad.txt"
        sfm_dir = output_dir / "sfm_superpoint+superglue"
        feature_path = output_dir / "feats-superpoint-n4096-r1024.h5"
        match_path = output_dir / "feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5"
        reference_model = root / scene / "sparse"
        # Step 3: Triangulation to obtain 3D model
        # model = reconstruction.main(sfm_dir, images_path, sfm_pairs, feature_path, match_path)
        model = triangulation.main(sfm_dir, reference_model, images_path, sfm_pairs, feature_path, match_path)
    print(f"Finished processing scene: {scene}.")