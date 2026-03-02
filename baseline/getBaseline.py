import argparse
import torch
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Setup Logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_h5_to_torch(path, key=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with h5py.File(path, 'r') as f:
        if key is None:
            if 'points3d' in f:
                data = f['points3d']['descriptors'].__array__()
            else:
                raise KeyError(f"Could not find 'points3d' in {path}")
        else:
            data = f[key].__array__()
    return torch.from_numpy(data).float().to(DEVICE)

def compute_metrics(predictions, gt_labels):
    # Identify valid ground truth
    valid_mask = gt_labels > -1
    
    num_matchable = valid_mask.sum().item()
    num_unmatchable = (~valid_mask).sum().item()
    
    # Check correctness
    correct_mask = (predictions == gt_labels) & valid_mask
    num_correct = correct_mask.sum().item()
    
    # Calculate Metrics
    recall = num_correct / num_matchable if num_matchable > 0 else 0.0
    num_predicted = len(predictions)
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    accuracy = precision

    return {
        "num_matchable": num_matchable,
        "num_unmatchable": num_unmatchable,
        "match_recall": recall,
        "match_precision": precision,
        "accuracy": accuracy,
        "average_precision": precision 
    }

def process_scene(scene_path, outputs_root):
    scene_name = scene_path.name
    scene_output_dir = outputs_root / scene_name
    
    paths = {
        "feats_2d": scene_output_dir / "feats-superpoint.h5",
        "feats_3d": scene_output_dir / "features_3d.h5",
        "labels":   scene_output_dir / "labels_3d.h5"
    }

    if not all(p.exists() for p in paths.values()):
        logger.warning(f"Skipping {scene_name}: Missing .h5 files.")
        return None

    try:
        desc_3d = load_h5_to_torch(paths["feats_3d"])
        desc_3d = torch.nn.functional.normalize(desc_3d, p=2, dim=1)
    except Exception as e:
        logger.error(f"Error loading 3D for {scene_name}: {e}")
        return None

    scene_metrics = {
        "num_matchable": [], "num_unmatchable": [], 
        "match_recall": [], "match_precision": [], 
        "accuracy": [], "average_precision": []
    }

    with h5py.File(paths["labels"], 'r') as f_labels:
        image_names = list(f_labels.keys())
        
        with h5py.File(paths["feats_2d"], 'r') as f_2d:
            for img_name in tqdm(image_names, desc=f"Scene {scene_name}", leave=False):
                if img_name not in f_2d: continue

                gt_labels = torch.from_numpy(f_labels[img_name].__array__()).long().to(DEVICE)
                desc_2d = torch.from_numpy(f_2d[img_name]['descriptors'].__array__()).float().to(DEVICE)
                
                if desc_2d.shape[0] == 256 and desc_2d.shape[1] != 256:
                    desc_2d = desc_2d.t()

                # Safety Clip
                min_len = min(len(gt_labels), len(desc_2d))
                if min_len == 0: continue
                gt_labels = gt_labels[:min_len]
                desc_2d = desc_2d[:min_len]

                # Normalize and match
                desc_2d = torch.nn.functional.normalize(desc_2d, p=2, dim=1)
                similarity = torch.matmul(desc_2d, desc_3d.t())
                _, predictions = torch.max(similarity, dim=1)

                # Compute metrics for this image
                m = compute_metrics(predictions, gt_labels)
                
                # Append to scene list
                for k, v in m.items():
                    scene_metrics[k].append(v)

    # Average over images in the scene
    averaged_metrics = {}
    for k, v in scene_metrics.items():
        if v: averaged_metrics[k] = np.mean(v)
        else: averaged_metrics[k] = 0.0
        
    return averaged_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--scene_list', type=Path, default=None)
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()

    scenes = []
    if args.scene_list and args.scene_list.exists():
        logger.info(f"Reading scenes from {args.scene_list}...")
        with open(args.scene_list, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            scenes.append(args.dataset / name)
    elif args.scene:
        scenes = [args.dataset / args.scene]
    else:
        scenes = sorted([p for p in args.dataset.iterdir() if p.is_dir()])

    logger.info(f"Found {len(scenes)} scenes to process.")

    final_results = {}

    for scene_path in tqdm(scenes, desc="Total Progress"):
        metrics = process_scene(scene_path, args.outputs)
        if metrics:
            for k, v in metrics.items():
                if k not in final_results: final_results[k] = []
                final_results[k].append(v)

    print("\n" + "="*40)
    print("FINAL TEST RESULTS (Averaged)")
    print("="*40)
    
    order = [
        "num_matchable", "num_unmatchable", 
        "match_recall", "match_precision", 
        "accuracy", "average_precision"
    ]
    
    for k in order:
        if k in final_results and final_results[k]:
            mean_val = np.mean(final_results[k])
            print(f"{k:.<35} {mean_val:.4f}")
        else:
            print(f"{k:.<35} N/A")
            
    print("="*40)

if __name__ == "__main__":
    main()