import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import struct
from omegaconf import OmegaConf
from gluefactory.models.two_view_pipeline import TwoViewPipeline
from gluefactory.utils.tensor import batch_to_device
from gluefactory.datasets.megadepth_3d import MegaDepth3D

EXPERIMENT_ROOT = "/home/x_lishu/matching/glue-factory/outputs/training/lightglue_3d_experiment_v1"
CHECKPOINT_PATH = f"{EXPERIMENT_ROOT}/checkpoint_best.tar"
DATASET_ROOT = "/proj/vlarsson/datasets/megadepth/Undistorted_SfM" 
TRIANGULATION_OUTPUTS = "/proj/vlarsson/users/x_lishu/matching/outputs/triangulation"

OUTPUT_DIR = "outputs/visualizations"
NUM_BATCHES = 5
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def apply_quantile_filter(kpts_3d, matches_gt, matches_pred, q=0.05):
    if len(kpts_3d) == 0:
        return kpts_3d, matches_gt, matches_pred

    # Calculate bounds
    q_low = torch.quantile(kpts_3d, q, dim=0)
    q_high = torch.quantile(kpts_3d, 1 - q, dim=0)
    
    # Create mask
    mask_box = (kpts_3d >= q_low) & (kpts_3d <= q_high)
    valid_mask = mask_box.all(dim=1) # Keep point only if valid in X, Y, and Z
    
    # Filter points
    kpts_3d_new = kpts_3d[valid_mask]
    
    # Create Index Mapping (Old Index -> New Index)
    old_indices = torch.where(valid_mask)[0]
    lookup = torch.full((len(kpts_3d),), -1, device=kpts_3d.device, dtype=torch.long)
    lookup[old_indices] = torch.arange(len(old_indices), device=kpts_3d.device)
    
    # Remap Ground Truth Matches
    new_matches_gt = matches_gt.clone()
    valid_gt = matches_gt > -1
    # Remap valid indices. If a point was filtered out, lookup returns -1 (becoming unmatched)
    new_matches_gt[valid_gt] = lookup[matches_gt[valid_gt]]
    
    # Remap Predicted Matches
    new_matches_pred = matches_pred.clone()
    valid_pred = matches_pred > -1
    new_matches_pred[valid_pred] = lookup[matches_pred[valid_pred]]
    
    return kpts_3d_new, new_matches_gt, new_matches_pred

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def get_image_pose(scene_path, image_name):
    model_paths = [
        scene_path / "sparse/0/images.bin",
        scene_path / "colmap/sparse/0/images.bin",
        scene_path / "sparse/images.bin"
    ]
    bin_path = None
    for p in model_paths:
        if p.exists():
            bin_path = p
            break
    if bin_path is None: return None, None

    img_name_lookup = Path(image_name).name
    with open(bin_path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            image_name_bytes = b""
            while True:
                char = fid.read(1)
                if char == b"\x00": break
                image_name_bytes += char
            current_name = image_name_bytes.decode("utf-8")
            if Path(current_name).name == img_name_lookup:
                return qvec2rotmat(qvec), tvec
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1)
    return None, None

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

def plot_validation_3d(image, kpts_2d, kpts_3d, pred_matches, gt_matches, save_path, scene_id, R=None, t=None):
    kpts_2d = kpts_2d.cpu().numpy()
    kpts_3d = kpts_3d.cpu().numpy()
    pred = pred_matches.cpu().numpy()
    gt = gt_matches.cpu().numpy()

    # Correct (Green): Predicted a match, and it equals GT
    mask_correct = (pred > -1) & (pred == gt)
    
    # Wrong (Red): Predicted a match, but it is NOT the GT (or GT was -1)
    mask_wrong = (pred > -1) & (pred != gt)
    
    # Missed (Blue): Predicted NO match (-1), but GT had a match (> -1)
    mask_missed = (pred == -1) & (gt > -1)

    # Transform 3D points
    if R is not None and t is not None:
        kpts_3d = (np.dot(R, kpts_3d.T) + t[:, None]).T
        pose_text = "View: Camera-Aligned"
    else:
        pose_text = "View: World Frame"

    fig = plt.figure(figsize=(24, 12))

    # Left: 2D image
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.imshow(image)
    
    if mask_missed.any():
        ax2d.scatter(kpts_2d[mask_missed, 0], kpts_2d[mask_missed, 1], 
                     c='blue', s=20, alpha=0.6, label='Missed (FN)')
    if mask_wrong.any():
        ax2d.scatter(kpts_2d[mask_wrong, 0], kpts_2d[mask_wrong, 1], 
                     c='red', s=30, marker='x', linewidths=2, label='Wrong (FP)')
    if mask_correct.any():
        ax2d.scatter(kpts_2d[mask_correct, 0], kpts_2d[mask_correct, 1], 
                     c='lime', s=40, marker='o', edgecolors='black', label='Correct (TP)')

    ax2d.legend(loc='upper right', fontsize=12)
    ax2d.set_title(f"2D query image: {mask_correct.sum()} Correct | {mask_wrong.sum()} Wrong | {mask_missed.sum()} Missed", fontsize=16)
    ax2d.axis('off')

    # Right: 3D pointcloud
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    # Background Structure (Gray)
    ax3d.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2], 
                 c='gray', s=10, alpha=0.3, label='Structure')

    # Correct Matches in 3D (Green)
    valid_correct_indices = pred[mask_correct]
    # Filter out indices that might have been quantile-filtered
    valid_correct_indices = valid_correct_indices[valid_correct_indices < len(kpts_3d)]
    
    if len(valid_correct_indices) > 0:
        pts_correct = kpts_3d[valid_correct_indices]
        ax3d.scatter(pts_correct[:, 0], pts_correct[:, 1], pts_correct[:, 2], 
                     c='lime', s=40, marker='^', edgecolors='black', label='Correct 3D')

    # Wrong Matches in 3D (Red)
    valid_wrong_indices = pred[mask_wrong]
    valid_wrong_indices = valid_wrong_indices[valid_wrong_indices < len(kpts_3d)]
    
    if len(valid_wrong_indices) > 0:
        pts_wrong = kpts_3d[valid_wrong_indices]
        ax3d.scatter(pts_wrong[:, 0], pts_wrong[:, 1], pts_wrong[:, 2], 
                     c='red', s=40, marker='v', edgecolors='black', label='Wrong 3D')

    ax3d.set_title(f"3D pointcloud ({pose_text})\nFiltered (5%-95%)", fontsize=16)
    
    if R is not None:
        ax3d.view_init(elev=-90, azim=-90)

    # Set Axes Equal
    x_limits = ax3d.get_xlim3d()
    y_limits = ax3d.get_ylim3d()
    z_limits = ax3d.get_zlim3d()
    r = 0.5 * max([abs(x_limits[1]-x_limits[0]), abs(y_limits[1]-y_limits[0]), abs(z_limits[1]-z_limits[0])])
    x_c, y_c, z_c = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax3d.set_xlim3d([x_c-r, x_c+r])
    ax3d.set_ylim3d([y_c-r, y_c+r])
    ax3d.set_zlim3d([z_c-r, z_c+r])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)
    
    conf = {
        'name': 'megadepth_3d',
        'root': TRIANGULATION_OUTPUTS,
        'output_dir': 'outputs',
        'train_split': 'dummy', 
        'val_split': '/home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/test_scenes_clean_try.txt',
        'test_split': 'dummy',
        'batch_size': BATCH_SIZE,
        'num_workers': 1,
        'max_num_points3d': 2048, 
        'max_num_keypoints': 2048,
        'force_num_points3d': True,
        'seed': 0,
    }
    
    dataset = MegaDepth3D(OmegaConf.create(conf))
    loader = dataset.get_data_loader("val")
    
    print(f"Starting Validation Visualization (with Quantile Filtering)...")
    
    for i, data in enumerate(loader):
        if i >= NUM_BATCHES: break
        
        data = batch_to_device(data, DEVICE)
        with torch.no_grad():
            pred = model(data)
        
        batch_size_curr = len(data["view0"]["image_name"])
        for b in range(batch_size_curr):
            img_rel_path = data["view0"]["image_name"][b] 
            scene_id = data["scene"][b]
            
            img_path = Path(DATASET_ROOT) / scene_id / "images" / Path(img_rel_path).name
            if not img_path.exists(): img_path = Path(DATASET_ROOT) / img_rel_path    
            if not img_path.exists(): continue
                
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pred_matches = pred["matches0"][b]
            gt_matches = data["matches0"][b]
            kpts_2d = data["view0"]["keypoints"][b]
            kpts_3d = data["view1"]["keypoints"][b]
            
            kpts_3d, gt_matches, pred_matches = apply_quantile_filter(kpts_3d, gt_matches, pred_matches, q=0.05)
            
            scene_path = Path(DATASET_ROOT) / scene_id
            try:
                R, t = get_image_pose(scene_path, img_rel_path)
            except:
                R, t = None, None
            
            safe_name = Path(img_rel_path).stem
            save_path = Path(OUTPUT_DIR) / f"{scene_id}_{safe_name}.png"
            
            plot_validation_3d(image, kpts_2d, kpts_3d, pred_matches, gt_matches, save_path, scene_id, R, t)

if __name__ == "__main__":
    main()