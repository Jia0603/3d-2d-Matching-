import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from hloc import triangulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_scene(scene_path, output_path):
    scene_name = scene_path.name
    logger.info(f"Processing Scene (CPU Step): {scene_name}...")
    
    images_dir = scene_path / "images"
    scene_output_dir = output_path / scene_name

    pairs_path = scene_output_dir / "pairs-netvlad.txt"
    local_feats = scene_output_dir / "feats-superpoint.h5"
    matches_path = scene_output_dir / "matches-superpoint-lightglue.h5"
    
    if not (pairs_path.exists() and local_feats.exists() and matches_path.exists()):
        logger.warning(f"Skipping {scene_name}: Missing features/matches.")
        return

    possible_paths = [
        scene_path / "sparse" / "manhattan" / "0",
        scene_path / "sparse" / "0",
        scene_path / "sparse" 
    ]
    reference_sfm = None
    for p in possible_paths:
        if p.exists() and (p / "cameras.bin").exists():
            reference_sfm = p
            break
    if reference_sfm is None:
        logger.warning(f"Skipping {scene_name}: No sparse model found.")
        return

    sfm_output = scene_output_dir / "sfm_superpoint_triangulated"
    if (sfm_output / "points3D.bin").exists():
        logger.info(f"Skipping {scene_name}: Output already exists at {sfm_output}")
        return

    # Run Triangulation
    triangulation.main(
        sfm_output,
        reference_sfm,
        images_dir,
        pairs_path,
        local_feats,
        matches_path,
        skip_geometric_verification=True
    )
    
    logger.info(f"Successfully triangulated {scene_name}. Results in {sfm_output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--scene_list', type=Path, default=None)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset}")

    scenes = []
    if args.scene_list:
        if not args.scene_list.exists():
            raise FileNotFoundError(f"Scene list file not found: {args.scene_list}")
        logger.info(f"Reading scenes from {args.scene_list}...")
        with open(args.scene_list, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        for name in scene_names:
            scenes.append(args.dataset / name)
    elif args.scene:
        scenes = [args.dataset / args.scene]
    else:
        scenes = sorted([p for p in args.dataset.iterdir() if p.is_dir() and (p / "images").exists()])

    logger.info(f"Found {len(scenes)} scenes to process.")

    for scene_path in tqdm(scenes):
        try:
            process_scene(scene_path, args.outputs)
        except Exception as e:
            logger.error(f"Failed to process {scene_path.name}: {e}")
            continue

if __name__ == "__main__":
    main()