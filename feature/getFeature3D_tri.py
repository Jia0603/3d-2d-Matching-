import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
from hloc.utils.read_write_model import read_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_3d_descriptors(scene_path, output_path):
    scene_name = scene_path.name
    logger.info(f"Processing Scene (3D Descriptors): {scene_name}...")

    scene_output_dir = output_path / scene_name
    sfm_model_dir = scene_output_dir / "sfm_superpoint_triangulated"
    feats_2d_path = scene_output_dir / "feats-superpoint.h5"
    output_3d_path = scene_output_dir / "features_3d.h5"

    if not sfm_model_dir.exists():
        logger.warning(f"Skipping {scene_name}: No SFM model found at {sfm_model_dir}")
        return
    
    if output_3d_path.exists():
        logger.info(f"Skipping {scene_name}: features_3d.h5 already exists.")
        return

    logger.info("Loading COLMAP model...")
    cameras, images, points3D = read_model(sfm_model_dir)
    
    logger.info("Loading 2D descriptors...")
    
    with h5py.File(feats_2d_path, 'r') as f:
        # Map image_name -> descriptors
        # Only load descriptors
        descriptors_map = {}
        for img_name in f.keys():
            desc = f[img_name]['descriptors'].__array__()
            if desc.shape[0] == 256 and desc.shape[1] != 256:
                desc = desc.T
            descriptors_map[img_name] = desc

    # Compute 3D descriptors
    logger.info("Computing averaged 3D descriptors...")
    
    xyz_list = []
    desc_list = []
    
    # Iterate over every 3D point
    for p3d_id, p3d in tqdm(points3D.items()):
        xyz_list.append(p3d.xyz)
        
        track_descriptors = []
        
        for img_id, point2d_idx in zip(p3d.image_ids, p3d.point2D_idxs):
            img_name = images[img_id].name
            if img_name in descriptors_map:
                # Get the specific descriptor for that keypoint
                d = descriptors_map[img_name][point2d_idx]
                track_descriptors.append(d)
        
        if len(track_descriptors) == 0:
            desc_list.append(np.zeros(256, dtype=np.float32))
            continue

        # Average them
        track_descriptors = np.array(track_descriptors)
        mean_desc = np.mean(track_descriptors, axis=0)

        # L2 normalize
        norm = np.linalg.norm(mean_desc)
        if norm > 1e-6:
            mean_desc /= norm
            
        desc_list.append(mean_desc)

    # Save
    logger.info(f"Saving {len(xyz_list)} points to {output_3d_path}...")
    
    with h5py.File(output_3d_path, 'w') as f:
        grp = f.create_group("points3d")
        grp.create_dataset("xyz", data=np.array(xyz_list, dtype=np.float32))
        grp.create_dataset("descriptors", data=np.array(desc_list, dtype=np.float32))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()

    if args.scene:
        scenes = [args.outputs / args.scene]
    else:
        scenes = sorted([p for p in args.outputs.iterdir() if p.is_dir()])

    logger.info(f"Found {len(scenes)} scenes to process.")

    for scene_path in scenes:
        try:
            compute_3d_descriptors(scene_path, args.outputs)
        except Exception as e:
            logger.error(f"Failed to process {scene_path.name}: {e}")
            continue

if __name__ == "__main__":
    main()