from pathlib import Path
from visual_sfm_3d import visualize_sfm_3d
from hloc import (
    extract_features,
    match_features,
    pairs_from_covisibility,
)
from hloc import colmap_from_nvm, triangulation
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
dataset = Path("/proj/vlarsson/datasets/aachen_v1.1/")  # change this if your dataset is somewhere else
images = dataset / "images/images_upright/"
html_save_dir =Path("/home/x_jiagu/degree_project/SfM_htmls")

outputs = Path("/proj/vlarsson/outputs_aachen_origin/sfm/")  # where everything will be saved
sfm_pairs = outputs / "pairs-db-covis20.txt"  # top 20 most covisible in SIFT model
# loc_pairs = outputs / "pairs-query-netvlad20.txt"  # top 20 retrieved by NetVLAD
reference_sfm = outputs / "sfm_superpoint+lightglue"  # the SfM model we will build
results = outputs / "Aachen_hloc_superpoint+lightglue_netvlad20.txt"  # the result file

feature_conf = {
    'output': 'feats-superpoint-n2048',
    'model': {
        'name': 'superpoint',
        'nms_radius': 4,
        'max_keypoints': 2048, 
    },
    'preprocessing': {
        'grayscale': True,
        'resize_max': 1024, 
        "resize_force": False,
    },
}

if device.type == "cuda":
    print("Using GPU for processing feature extraction and matching.")
    # retrieval_conf = extract_features.confs["netvlad"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    features = extract_features.main(feature_conf, images, outputs)

    colmap_from_nvm.main(
        dataset / "3D-models/aachen_cvpr2018_db.nvm",
        dataset / "3D-models/database_intrinsics.txt",
        dataset / "aachen.db",
        outputs / "sfm_sift",
    )

    pairs_from_covisibility.main(outputs / "sfm_sift", sfm_pairs, num_matched=20)

    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
else:
    print("Using CPU for processing 3D triangulation.")
    ## below is CPU step for triangulation
    features = outputs / "feats-superpoint-n2048.h5"
    sfm_matches = outputs / "feats-superpoint-n2048_matches-superpoint-lightglue_pairs-db-covis20.h5"
    reconstruction = triangulation.main(
        reference_sfm, outputs / "sfm_sift", images, sfm_pairs, features, sfm_matches
    )

    visualize_sfm_3d(reference_sfm, "aachen", html_save_dir, True)