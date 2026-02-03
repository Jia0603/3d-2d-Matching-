from pathlib import Path
from visual_sfm_3d import visualize_sfm_3d
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from hloc import triangulation

root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
outputs = Path("/proj/vlarsson/outputs/sfm/")
html_save_dir =Path("/home/x_jiagu/degree_project/SfM_htmls")

overlap_thres = [0.3, 0.95]  # define your overlap threshold range here

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

scene_names = sorted([
    p.name
    for p in root.iterdir()
    if p.is_dir()
])

for scene in scene_names[:12]: # change the slice to process more scenes
    print(f"Start processing scene: {scene}...")

    images_path = root / scene / "images"

    output_dir = outputs / scene
    output_dir.mkdir(parents=True, exist_ok=True)

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
    visualize_sfm_3d(sfm_dir, scene, html_save_dir)
