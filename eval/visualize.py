import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from omegaconf import OmegaConf
from gluefactory.models.two_view_pipeline import TwoViewPipeline
from gluefactory.utils.tensor import batch_to_device
from gluefactory.datasets.megadepth_3d import MegaDepth3D

EXPERIMENT_ROOT = "/home/x_lishu/matching/glue-factory/outputs/training/lightglue_3d_experiment_v1"
CHECKPOINT_PATH = f"{EXPERIMENT_ROOT}/checkpoint_best.tar"
DATASET_ROOT = "/proj/vlarsson/datasets/megadepth/Undistorted_SfM" 
TRIANGULATION_OUTPUTS = "/proj/vlarsson/users/x_lishu/matching/outputs/triangulation"

OUTPUT_DIR = "outputs/visualizations"
NUM_BATCHES = 5   # How many batches to visualize
BATCH_SIZE = 4    # Images per batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    print(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=DEVICE)
    conf = checkpoint["conf"]["model"]
    conf["allow_no_extract"] = True
    
    model = TwoViewPipeline(conf)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model

def plot_matches(image, kpts, matches, save_path, title="Matches"):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    kpts = kpts.cpu().numpy()
    matches = matches.cpu().numpy()
    
    valid = matches > -1
    mkpts = kpts[valid]
    unmatched_kpts = kpts[~valid]
    
    # Plot unmatched points (Red dots)
    plt.scatter(unmatched_kpts[:, 0], unmatched_kpts[:, 1], c='r', s=2, alpha=0.4, label='Unmatched')
    # Plot matched points (Green crosses)
    plt.scatter(mkpts[:, 0], mkpts[:, 1], c='lime', s=15, marker='x', linewidths=1.5, label='Matched to 3D')
    
    plt.title(f"{title} ({len(mkpts)} matches)")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved visualization to {save_path}")

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)
    
    conf = {
        'name': 'megadepth_3d',
        'root': TRIANGULATION_OUTPUTS, # .h5 files
        'output_dir': 'outputs',
        'train_split': 'dummy', 
        'val_split': '/home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/test_scenes_clean_try.txt',
        'test_split': 'dummy',
        'max_num_points3d': 2048,
        'max_num_keypoints': 2048,
        'force_num_points3d': True,
        'seed': 0,
        'batch_size': BATCH_SIZE,
        'num_workers': 1,
    }
    
    dataset = MegaDepth3D(OmegaConf.create(conf))
    loader = dataset.get_data_loader("val")
    
    print(f"Starting visualization of {NUM_BATCHES} batches...")
    
    # Visualization
    for i, data in enumerate(loader):
        if i >= NUM_BATCHES: 
            print("Reached batch limit.")
            break 
        
        data = batch_to_device(data, DEVICE)
        
        with torch.no_grad():
            pred = model(data)
        
        # Loop over images in the batch
        batch_size_curr = len(data["view0"]["image_name"])
        
        for b in range(batch_size_curr):
            img_rel_path = data["view0"]["image_name"][b] 
            scene_id = data["scene"][b]
            
            # Construct raw image path
            img_path = Path(DATASET_ROOT) / scene_id / "images" / Path(img_rel_path).name
            if not img_path.exists():
                img_path = Path(DATASET_ROOT) / img_rel_path    
            
            if not img_path.exists():
                print(f"Could not find image at {img_path}. Skipping.")
                continue
                
            # Load Raw Image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to read image file: {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            matches = pred["matches0"][b]
            kpts = data["view0"]["keypoints"][b]
            
            safe_name = Path(img_rel_path).stem
            save_path = Path(OUTPUT_DIR) / f"vis_{scene_id}_{safe_name}.png"
            
            plot_matches(image, kpts, matches, save_path, title=f"Scene {scene_id} - {safe_name}")

if __name__ == "__main__":
    main()