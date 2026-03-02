import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import logging
import sys

from gluefactory.models.two_view_pipeline import TwoViewPipeline
from gluefactory.datasets.megadepth_3d import MegaDepth3D
from gluefactory.utils.tensor import batch_to_device

EXPERIMENT_NAME = "lightglue_3d_experiment_v1"
CHECKPOINT_PATH = f"/home/x_lishu/matching/glue-factory/outputs/training/{EXPERIMENT_NAME}/checkpoint_best.tar"
TEST_LIST = "/home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/test_scenes_clean_try.txt"
DATASET_ROOT = "/proj/vlarsson/users/x_lishu/matching/outputs/triangulation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    conf = ckpt["conf"]["model"]
    conf["allow_no_extract"] = True
    
    print("Initializing Model...")
    model = TwoViewPipeline(conf)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()

    data_conf = {
        'name': 'megadepth_3d',
        'root': DATASET_ROOT,  
        'output_dir': '/home/x_lishu/matching/glue-factory/outputs', 
        'train_split': 'dummy_train', 
        'val_split': 'dummy_val',
        'test_split': TEST_LIST,
        'batch_size': 4,
        'num_workers': 4,
        'max_num_points3d': 2048,
        'max_num_keypoints': 2048, 
        'force_num_points3d': True,
        'seed': 42,
    }
    
    print("Initializing Dataset...")
    dataset = MegaDepth3D(OmegaConf.create(data_conf))
    
    loader = dataset.get_data_loader("test")
    
    if len(loader) == 0:
        print("Error: Test loader is empty. Check your test_scenes_clean_try.txt path.")
        return

    print(f"Starting evaluation on {len(loader)} batches...")
    
    results = {}
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = batch_to_device(data, DEVICE)
            pred = model(data)
            losses, metrics = model.loss(pred, {**data, **pred})
            all_stats = {**losses, **metrics}
            
            for k, v in all_stats.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() > 1:
                        v = v.mean() 
                    v = v.item()
                
                if k not in results: 
                    results[k] = []
                results[k].append(v)

    print("\n" + "="*40)
    print(f"FINAL RESULTS: {EXPERIMENT_NAME}")
    print("="*40)
    
    priority_keys = [
        "match_recall", "match_precision", "accuracy", 
        "num_matchable", "num_unmatchable",
        "confidence", "total"
    ]
    
    for k in priority_keys:
        if k in results:
            valid_vals = [x for x in results[k] if np.isfinite(x)]
            if valid_vals:
                print(f"{k:.<35} {np.mean(valid_vals):.4f}")
    
    for k in sorted(results.keys()):
        if k not in priority_keys:
            valid_vals = [x for x in results[k] if np.isfinite(x)]
            if valid_vals:
                print(f"{k:.<35} {np.mean(valid_vals):.4f}")
                
    print("="*40)

if __name__ == "__main__":
    evaluate()