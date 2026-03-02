import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from hloc.utils.read_write_model import read_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_labels(scene_path, output_path):
    scene_name = scene_path.name
    logger.info(f"Generating Labels for: {scene_name}...")

    scene_output_dir = output_path / scene_name
    sfm_model_dir = scene_output_dir / "sfm_superpoint_triangulated"
    output_labels_path = scene_output_dir / "labels_3d.h5"
    
    if not sfm_model_dir.exists():
        logger.warning(f"Skipping {scene_name}: No SFM model.")
        return
    if output_labels_path.exists():
        logger.info(f"Skipping {scene_name}: Labels already exist.")
        return

    logger.info("Loading COLMAP geometry...")
    cameras, images, points3D = read_model(sfm_model_dir)

    # Build a Map: COLMAP_ID -> Index in features_3d.h5
    logger.info("Building ID map...")
    p3d_id_to_index = {}
    current_index = 0
    for p_id in points3D:
        p3d_id_to_index[p_id] = current_index
        current_index += 1
    
    # Create Labels for every Image
    logger.info(f"Creating labels for {len(images)} images...")
    
    with h5py.File(output_labels_path, 'w') as f:
        for img_id, img in tqdm(images.items()):
            image_name = img.name
            
            # Get the list of 3D points visible in this image
            # Values are COLMAP_IDs, or -1 if not triangulated
            colmap_ids = img.point3D_ids
            
            # Convert COLMAP_IDs to 0..N indices
            label_indices = np.full(colmap_ids.shape, -1, dtype=np.int32)
            
            for i, cid in enumerate(colmap_ids):
                if cid != -1 and cid in p3d_id_to_index:
                    label_indices[i] = p3d_id_to_index[cid]

            f.create_dataset(image_name, data=label_indices)

    logger.info(f"Saved labels to {output_labels_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--scene_list', type=Path, default=None)
    args = parser.parse_args()

    scenes = []
    if args.scene_list:
        with open(args.scene_list, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            scenes.append(args.outputs / name)
    elif args.scene:
        scenes = [args.outputs / args.scene]
    else:
        scenes = sorted([p for p in args.outputs.iterdir() if p.is_dir()])

    for scene_path in scenes:
        try:
            generate_labels(scene_path, args.outputs)
        except Exception as e:
            logger.error(f"Failed {scene_path.name}: {e}")

if __name__ == "__main__":
    main()