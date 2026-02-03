import numpy as np
from hloc.utils import read_write_model as rw
import os
import random
from pathlib import Path

random.seed(42)
def extract_query_sets(scene: str, 
                       sfm_model_dir="/proj/vlarsson/datasets/megadepth/Undistorted_SfM", 
                       output_dir="/proj/vlarsson/outputs/query_sets",
                       sample_ratio=0.001, 
                       query_image_ratio=0.2,
                       ):
    
    print(f"Processing scene: {scene}")

    sfm_model_path = Path(sfm_model_dir) / scene / "sparse"
    # print(f"Reading SfM model from: {sfm_model_path}")
    # print(f"Sample ratio: {sample_ratio}")
    # print(f"Query image ratio: {query_image_ratio}")

    output_path = Path(output_dir) / scene
    output_path.mkdir(parents=True, exist_ok=True)

    # read the SfM model
    cameras, images, points3D = rw.read_model(sfm_model_path, ext=".bin")
    point_ids = list(points3D.keys())
    total_points = len(point_ids)
    num_samples = int(total_points * sample_ratio)
    sampled_ids = random.sample(point_ids, num_samples)
    print(f"Original 3D points: {total_points}; Sampled 3D points: {num_samples}")

    # collect image names that observe the sampled 3D points
    query_image_names = set()
    oberved_image_ids = set()
    for pid in sampled_ids:
        oberved_image_ids.update(points3D[pid].image_ids)

    print(f"Total oberved images: {len(oberved_image_ids)}")

    # randomly select a subset of oberved images as query images
    maximum_query_images = int(len(oberved_image_ids) * query_image_ratio)
    query_image_ids = random.sample(
        list(oberved_image_ids), 
        min(len(oberved_image_ids), maximum_query_images)
        )

    print(f"Query images collected: {len(query_image_ids)}")

    query_infos = []
    for img_id in query_image_ids:
        image = images[img_id]
        camera = cameras[image.camera_id]
        camera_info = f"{camera.id} {camera.model} {camera.width} {camera.height} " + \
                      " ".join(map(str, camera.params))
        query_infos.append((image.name, camera_info))

    # write query image names and camera infos to files
    query_name_path = output_path / "query_image_names.txt"
    query_camera_path = output_path / "query_image_cameras.txt"

    with open(query_name_path, "w") as f_n, open(query_camera_path, "w") as f_c:
        for name, camera_info in query_infos:
            f_n.write(f"{name}\n")
            f_c.write(f"{name} {camera_info}\n")

    print(f"Query image names written to: {query_name_path}")
    print(f"Query image cameras written to: {query_camera_path}")

if __name__ == "__main__":

    root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
    scene_names = sorted([p.name for p in root.iterdir() if p.is_dir()])

    for scene in scene_names[4:5]:  # change the slice to process more scenes
        extract_query_sets(scene,
                           sfm_model_dir="/proj/vlarsson/datasets/megadepth/Undistorted_SfM",
                           output_dir="/proj/vlarsson/outputs/query_sets/",
                           sample_ratio=0.001,
                           query_image_ratio=0.2)
        