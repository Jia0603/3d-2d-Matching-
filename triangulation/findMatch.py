import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from hloc import extract_features, match_features, pairs_from_retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf_retrieval = extract_features.confs['netvlad']

conf_feature = {
    'model': {
        'name': 'superpoint',
        'nms_radius': 3,
        'max_keypoints': 2048, # Adjust for denser cloudpoints
        'keypoint_threshold': 0.005,
    },
    'output': 'feats-superpoint-n2048-r1024',
    'preprocessing': {
        'grayscale': True,
        'resize_max': 1024,
    },
}

conf_matcher = match_features.confs['superpoint+lightglue']

def process_scene(scene_path, output_path):
    scene_name = scene_path.name
    logger.info(f"Processing Scene (GPU Step): {scene_name}...")
    
    images_dir = scene_path / "images"
    
    scene_output_dir = output_path / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    retrieval_feats = scene_output_dir / "global-feats-netvlad.h5"
    pairs_path = scene_output_dir / "pairs-netvlad.txt"
    local_feats = scene_output_dir / "feats-superpoint.h5"
    matches_path = scene_output_dir / "matches-superpoint-lightglue.h5"

    # Extract NetVLAD features
    if not retrieval_feats.exists():
        extract_features.main(
            conf_retrieval,
            images_dir,
            feature_path=retrieval_feats
        )
    
    # Find top 20 similar images
    if not pairs_path.exists():
        pairs_from_retrieval.main(
            retrieval_feats,
            pairs_path,
            num_matched=20
        )

    # Extract SuperPoint Features
    if not local_feats.exists():
        extract_features.main(
            conf_feature,
            images_dir,
            feature_path=local_feats
        )

    # Match with LightGlue
    if not matches_path.exists():
        match_features.main(
            conf_matcher,
            pairs_path,
            features=local_feats,
            matches=matches_path
        )
    
    logger.info(f"Finished Matching for {scene_name}.")

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