from pathlib import Path
from visual_sfm_3d import visualize_sfm_3d
from hloc.utils import read_write_model as rw
from hloc import triangulation, reconstruction, colmap_from_nvm
import os

root = Path("/proj/vlarsson/datasets/cambridge")
outputs = Path("/proj/vlarsson/outputs_cambridge/sfm/")
html_save_dir =Path("/home/x_jiagu/degree_project/SfM_htmls")
html_save_dir.mkdir(parents=True, exist_ok=True)
scene_names = ["StMarysChurch", "KingsCollege", "OldHospital", "ShopFacade", "GreatCourt"]
scale = 1.875 # 1920/1024, the reference model given in Cambridge is triangulated at 1024px

for scene in scene_names[1:]:
    print(f"Start processing scene: {scene}...")

    images_path = root / scene

    output_dir = outputs / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_model_dir = output_dir / "model_sift_1920px"
    scaled_model_dir.mkdir(parents=True, exist_ok=True)

    print("Using CPU for processing 3D triangulation.")
    sfm_pairs = output_dir / "pairs-covisib20.txt"
    sfm_dir = output_dir / "sfm_superpoint+lightglue"
    feature_path = output_dir / "feats-superpoint-n2048.h5"
    match_path = output_dir / "feats-superpoint-n2048_matches-superpoint-lightglue_pairs-covisib20.h5"
    reference_model = root / "CambridgeLandmarks_Colmap_Retriangulated_1024px" / scene / "model_train"

    print(f"Loading reference model from {reference_model}")
    cameras, images, points3D = rw.read_model(reference_model, ext='.bin')

    for cam_id, cam in cameras.items():
        new_params = cam.params.copy()
        if cam.model == "SIMPLE_RADIAL":
            new_params[0:3] *= scale 
        elif cam.model == "PINHOLE":
            new_params[0:4] *= scale
        
        cameras[cam_id] = cam._replace(
            params=new_params, 
            width=1920, 
            height=1080
        )
    for img_id, img in images.items():
        new_xys = img.xys * scale
        images[img_id] = img._replace(xys=new_xys)
    
    rw.write_model(cameras, images, points3D, scaled_model_dir, ext=".bin")

    # Triangulation to obtain 3D model
    model = triangulation.main(sfm_dir, scaled_model_dir, images_path,sfm_pairs, feature_path, match_path)
    
    os.remove(match_path)
    print(f"Removed intermediate files: {match_path}.")
    print(f"Secene {scene} 3D triangulation on CPU completed.")

    # Visualization
    visualize_sfm_3d(sfm_dir, scene, html_save_dir, True)
