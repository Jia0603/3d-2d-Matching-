import argparse
import logging
import pickle
import random
import numpy as np
import torch
import h5py
from pathlib import Path
from PIL import Image
import pycolmap
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from ground_truth.generate_gt_pairs_re import load_query_cams, compute_ground_truth_matches
from lightglue import LightGlue
from .visualize_nn_matches import MockCamera, get_most_similar_ref, launch_rerun_visualization
import matplotlib.pyplot as plt
from lightglue import viz2d
from lightglue.utils import rbd
from baseline.rr_baseline import compute_rr_baseline

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visual_flat_sfm(res, q_kpts, p3d_flat_kpts, img_query, p3d_colors, flat_w, flat_h, scene):
    logger.info("Generating 2D Flat SfM visualization...")
    
    # Create colored flat SfM image
    flat_img = np.ones((flat_h, flat_w, 3), dtype=np.float32)
    
    # Paint the 3D point colors onto the 2D image
    for (x, y), c in zip(p3d_flat_kpts.astype(int), p3d_colors):
        if 0 <= y < flat_h and 0 <= x < flat_w:
            flat_img[y, x] = c

    # Convert to format expected by LightGlue viz2d
    img_q_tensor = torch.from_numpy(img_query).float().permute(2, 0, 1)
    img_flat_tensor = torch.from_numpy(flat_img).float().permute(2, 0, 1)

    # Strip batch dimensions
    res_rbd = rbd(res)
    matches = res_rbd["matches"].cpu().numpy()
    
    m_kpts0 = q_kpts[matches[..., 0]]
    m_kpts1 = p3d_flat_kpts[matches[..., 1]]

    # The Matches plot
    viz2d.plot_images([img_q_tensor, img_flat_tensor])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {res_rbd["stop"]} layers', fs=20)

    # Lock bounds and prevent stretching
    axes = plt.gcf().axes
    if len(axes) >= 2:
        h0, w0 = img_query.shape[:2]
        axes[0].set_xlim(0, w0)
        axes[0].set_ylim(h0, 0)
        axes[1].set_xlim(0, flat_w)
        axes[1].set_ylim(flat_h, 0)

    match_filename = f"flat_matches_scene_{scene}.png"
    plt.savefig(match_filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

    # The pruning plot
    if "prune0" in res_rbd:
        kpc0 = viz2d.cm_prune(res_rbd["prune0"])
        kpc1 = viz2d.cm_prune(res_rbd["prune1"])
        viz2d.plot_images([img_q_tensor, img_flat_tensor])
        viz2d.plot_keypoints([torch.from_numpy(q_kpts), torch.from_numpy(p3d_flat_kpts)], colors=[kpc0, kpc1], ps=6)
        
        # Lock bounds and prevent stretching
        axes = plt.gcf().axes
        if len(axes) >= 2:
            axes[0].set_xlim(0, w0)
            axes[0].set_ylim(h0, 0)
            axes[1].set_xlim(0, flat_w)
            axes[1].set_ylim(flat_h, 0)
            
        prune_filename = f"flat_pruning_scene_{scene}.png"
        plt.savefig(prune_filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
    logger.info(f"Saved 2D Flat SfM images: {match_filename} & {prune_filename}")

def main():
    parser = argparse.ArgumentParser(description="Visualize RR Baseline Matches in Rerun")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth maps")
    parser.add_argument('--scene', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = args.scene
    logger.info(f"Starting visualization for scene {scene}...")

    # Select a random query and its reference
    query_names_file = args.query_dir / scene / "query_image_names.txt"
    with open(query_names_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    query_name = random.choice(queries)
    ref_name = get_most_similar_ref(query_name, args.covisibility_dir / scene / "most_similar_pairs.txt")
    logger.info(f"Query: {query_name} | Reference: {ref_name}")

    # Load query camera info
    query_cams = load_query_cams(args.query_dir / scene / "query_image_cameras.txt")
    q_camera = query_cams[query_name]
    q_img_size = [q_camera["intrinsics"]["width"], q_camera["intrinsics"]["height"]]

    # Load query features
    with h5py.File(args.sfm_dir / scene / "feats-superpoint-n2048.h5", "r") as f:
        q_kpts = f[query_name]["keypoints"][:]
        q_desc = f[query_name]["descriptors"][:]

    # Load SfM and visible points
    sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)

    with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
        visible_p3d = pickle.load(f)[query_name]["unique_points"]
        
    p3d_desc, p3d_kpts, p3d_colors = [], [], []
    
    with h5py.File(args.covisibility_dir / scene / "points3D_feats_cache.h5", "r") as f:
        for pid in visible_p3d:
            pid_str = str(pid)
            pid_int = int(pid)
            if pid_str in f and pid_int in reconstruction.points3D:
                p3d_desc.append(f[pid_str]["descriptors"][:].reshape(256))
                p3d_kpts.append(f[pid_str]["keypoints"][:].reshape(3))
                p3d_colors.append(reconstruction.points3D[pid_int].color)
                
    if not p3d_kpts:
        logger.error("No valid 3D coordinates/features found.")
        return
                
    p3d_desc = np.vstack(p3d_desc).T 
    p3d_kpts = np.vstack(p3d_kpts)   
    p3d_colors = np.vstack(p3d_colors) / 255.0 
    raw_pts_np = p3d_kpts.copy() 
    raw_colors_np = np.vstack(p3d_colors)


    # Load reference camera pose
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
    ref_R = qvec2rotmat(ref_image_obj.qvec)
    ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))

    # Get predicted matches and the flat data
    logger.info("Initialize LightGlue...")
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    rr_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_rr_baseline(
        matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device
    )

    # Calculate ground truth
    with h5py.File(args.depth_dir / scene / f"{Path(query_name).stem}.h5", 'r') as f:
        depth_map = f['depth'][:]
        
    gt_matches0, _ = compute_ground_truth_matches(
        {"keypoints": q_kpts}, {"keypoints": p3d_kpts}, q_camera, depth_map
    )
    
    # Evaluate metrics
    valid_pred = rr_matches0 > -1
    valid_gt = gt_matches0 > -1
    correct_matches = (rr_matches0 == gt_matches0) & valid_gt

    num_pred = valid_pred.sum()
    num_gt = valid_gt.sum()
    num_correct = correct_matches.sum()

    precision = num_correct / num_pred if num_pred > 0 else 0
    recall = num_correct / num_gt if num_gt > 0 else 0

    logger.info("="*30)
    logger.info("Rotate+Remove_coord Baseline Results:")
    logger.info(f"GT Matches:        {num_gt}")
    logger.info(f"Predicted Matches: {num_pred}")
    logger.info(f"Correct Matches:   {num_correct}")
    logger.info(f"Precision:         {precision:.4f}")
    logger.info(f"Recall:            {recall:.4f}")
    logger.info("="*30)

    # Launch 2D flat images
    img_query = np.array(Image.open(args.dataset / scene / "images" / query_name).convert("RGB")) / 255.0
    visual_flat_sfm(res, q_kpts, p3d_flat_kpts, img_query, p3d_colors, flat_w, flat_h, scene)

    # Launch rerun
    launch_rerun_visualization(
        pred_matches0=rr_matches0, 
        gt_matches0=gt_matches0,  
        q_kpts=q_kpts, p3d_kpts=p3d_kpts, 
        raw_pts_np=raw_pts_np, raw_colors_np=raw_colors_np,
        scene=scene, args=args, query_name=query_name, ref_name=ref_name, 
        camera=q_camera,
        ref_pose_matrix=ref_pose_matrix,
        method_name="RR"
    )

if __name__ == "__main__":
    main()