# For baseline: NN(Nearest Neighbour), RR(Rotate+Remove_coord), 
#               RN(Rotate+Normalize), PR(Project to Reference)
# For train: TRAIN(Lightglu3d two self and one bidirectional cross), 
#            ADAPT(lightglue+adapter)

# Before use it, add the gluefactory path in the terminal
# export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

import argparse
import logging
import pickle
import random
import numpy as np
import torch
import h5py
import rerun as rr
from pathlib import Path
from PIL import Image
import pycolmap
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from . import rerun_johanna as rru 
from ground_truth.generate_gt_pairs_re import load_query_cams, compute_ground_truth_matches
from gluefactory.models import get_model
from lightglue import LightGlue
from baseline.pr_baseline import compute_pr_baseline
import matplotlib.pyplot as plt
from lightglue import viz2d
from lightglue.utils import rbd
from baseline.rr_baseline import compute_rr_baseline
from baseline.rn_baseline import compute_rn_baseline
from gluefactory.models.matchers.lightglu3d_bicross import LightGlu3D

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockCamera:
    def __init__(self, width, height, params):
        self.size = [width, height]
        self.f = [params[0], params[1]]
        self.c = [params[2], params[3]]

def get_most_similar_ref(query_name, pair_file_path):
    with open(pair_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == query_name:
                return parts[1]
    return None

def compute_nn_baseline(q_desc, p3d_desc, device):
    nn_conf = {
        "name": "matchers.nearest_neighbor_matcher",
        "mutual_check": True,
        "ratio_thresh": None,
        "distance_thresh": 0.75,
    }
    matcher = get_model(nn_conf["name"])(nn_conf).eval().to(device)

    data = {
        "descriptors0": torch.from_numpy(q_desc.T).unsqueeze(0).float().to(device),
        "descriptors1": torch.from_numpy(p3d_desc.T).unsqueeze(0).float().to(device)
    }

    with torch.no_grad():
        pred = matcher(data)
    
    pred_matches0 = pred['matches0'][0].cpu().numpy() 
    return pred_matches0

def load_trained_lightglu3d(checkpoint_path, device):
    logger.info(f"Loading trained LightGlu3D from {checkpoint_path}...")
    conf = {
        "name": "lightglu3d_bicross", 
        "input_dim": 256, 
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,
        "mp": False, 
        "depth_confidence": -1, 
        "width_confidence": -1, 
        "filter_threshold": 0.1, 
        "checkpointed": False,
    }
    matcher = LightGlu3D(conf).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) 
    matcher.load_state_dict(state_dict, strict=False)
    return matcher

def load_trained_adapt(checkpoint_path, device):
    logger.info(f"Loading trained Adapter from {checkpoint_path}...")
    conf = {
        "name": "lightglue_adapt", 
        "input_dim": 256, 
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,
        "mp": False, 
        "depth_confidence": -1, 
        "width_confidence": -1, 
        "filter_threshold": 0.1, 
        "checkpointed": False,
    }
    matcher = get_model(conf["name"])(conf).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) 
    matcher.load_state_dict(state_dict, strict=False)
    return matcher

def compute_trained_lightglu3d(matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device):
    data = {
        "keypoints0": torch.from_numpy(q_kpts).unsqueeze(0).float().to(device),
        "keypoints1": torch.from_numpy(p3d_kpts).unsqueeze(0).float().to(device),
        "descriptors0": torch.from_numpy(q_desc.T).unsqueeze(0).float().to(device), 
        "descriptors1": torch.from_numpy(p3d_desc.T).unsqueeze(0).float().to(device), 
        "view0": {
            "image_size": torch.tensor([q_img_size]).float().to(device)
        }
    }
    with torch.no_grad():
        pred = matcher(data)
    
    return pred['matches0'][0].cpu().numpy()


def launch_rerun_visualization(pred_matches0, gt_matches0, q_kpts, p3d_kpts, raw_pts_np, raw_colors_np, scene, args, query_name, ref_name, camera, ref_pose_matrix, method_name="Baseline", est_pose_matrix=None):
    logger.info(f"Initializing Rerun Analytics Dashboard for {method_name}...")
    
    # Initialize rerun
    query_stem = Path(query_name).stem
    rr.init(f"Matches_Scene_{scene}_{method_name}_{query_stem}", spawn=False)

    # Calculate matches info using the generic 'pred_matches0'
    valid_pred = pred_matches0 > -1
    valid_gt = gt_matches0 > -1
    
    idx_correct = valid_pred & valid_gt & (pred_matches0 == gt_matches0) # Correct prediction
    idx_confused = valid_pred & valid_gt & (pred_matches0 != gt_matches0) # Confused wrong prediction
    idx_hallucinated = valid_pred & (~valid_gt) # Hallucinated wrong prediction
    idx_missed = (~valid_pred) & valid_gt # Missed ground truth
    idx_unmatchable = (~valid_pred) & (~valid_gt) # Ignored unmatchable points

    # Set up cameras and images
    cameras, images, _ = rw.read_model(args.sfm_dir / scene / "sfm_superpoint+lightglue", ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
    ref_cam_obj = cameras[ref_image_obj.camera_id]
    ref_poselib_cam = MockCamera(ref_cam_obj.width, ref_cam_obj.height, ref_cam_obj.params)

    q_pose_matrix = np.hstack((qvec2rotmat(camera["qvec"]), np.array(camera["tvec"]).reshape(3, 1)))
    query_poselib_cam = MockCamera(camera["intrinsics"]["width"], camera["intrinsics"]["height"], camera["intrinsics"]["params"])

    img_query = np.array(Image.open(args.dataset / scene / "images" / query_name).convert("RGB")) / 255.0
    img_ref = np.array(Image.open(args.dataset / scene / "images" / ref_name).convert("RGB")) / 255.0

    rru.plot_scene(
        pts_3d=np.empty((0,3)), pts_2d=np.empty((0,2)),           
        img_query=img_query, imgs_refs=[img_ref], 
        camera_poses_refs=np.array([ref_pose_matrix]), 
        poselib_cam_intrinsics_q=query_poselib_cam,
        poselib_cam_intrinsics_refs=[ref_poselib_cam], 
        cam_pose_query_estimated=None,  
        cam_pose_query_gt=q_pose_matrix, 
        attach_image_to_est_pose=False  
    )

    # Load query with categorized keypoints
    img_path = "world/camera_query_gt/image"
    rr.log(f"{img_path}/Correct", rr.Points2D(q_kpts[idx_correct], colors=[0, 255, 0], radii=4.0)) # Green
    rr.log(f"{img_path}/Confused", rr.Points2D(q_kpts[idx_confused], colors=[255, 165, 0], radii=3.0)) # Orange
    rr.log(f"{img_path}/Hallucinated", rr.Points2D(q_kpts[idx_hallucinated], colors=[255, 0, 0], radii=3.0)) # Red
    rr.log(f"{img_path}/Missed", rr.Points2D(q_kpts[idx_missed], colors=[0, 150, 255], radii=3.0)) # Blue
    rr.log(f"{img_path}/Unmatchable", rr.Points2D(q_kpts[idx_unmatchable], colors=[128, 0, 128], radii=2.0)) # Purple

    # Load visible sfm model
    rr.log("world/SfM_Context", rr.Points3D(raw_pts_np, colors=raw_colors_np, radii=0.03))
    cam_center = (-q_pose_matrix[:, :3].T @ q_pose_matrix[:, 3]).flatten()

    # Correct predicted matches (Green points and lines)
    correct_3d_pts = p3d_kpts[pred_matches0[idx_correct]]
    correct_lines = [[cam_center, pt] for pt in correct_3d_pts]
    rr.log("world/Predictions/Correct/Points", rr.Points3D(correct_3d_pts, colors=[0, 255, 0], radii=0.06))
    rr.log("world/Predictions/Correct/Lines", rr.LineStrips3D(correct_lines, colors=[0, 255, 0, 100]))
    
    # Confused wrong prediction (Orange points and lines + yellow error vectors)
    confused_3d_pts = p3d_kpts[pred_matches0[idx_confused]]
    confused_lines = [[cam_center, pt] for pt in confused_3d_pts]
    rr.log("world/Predictions/Confused/Points", rr.Points3D(confused_3d_pts, colors=[255, 165, 0], radii=0.06))
    rr.log("world/Predictions/Confused/Lines", rr.LineStrips3D(confused_lines, colors=[255, 165, 0, 80]))
    pred_pts_for_error = p3d_kpts[pred_matches0[idx_confused]]
    gt_pts_for_error = p3d_kpts[gt_matches0[idx_confused]]
    error_lines = [[gt_pt, pred_pt] for gt_pt, pred_pt in zip(gt_pts_for_error, pred_pts_for_error)]
    rr.log("world/Predictions/Confused/Error_Vectors", rr.LineStrips3D(error_lines, colors=[255, 255, 0, 200])) 

    # Hallucinated wrong prediction (Red points and lines)
    hallucinated_3d_pts = p3d_kpts[pred_matches0[idx_hallucinated]]
    hallucinated_lines = [[cam_center, pt] for pt in hallucinated_3d_pts]
    rr.log("world/Predictions/Hallucinated/Points", rr.Points3D(hallucinated_3d_pts, colors=[255, 0, 0], radii=0.06))
    rr.log("world/Predictions/Hallucinated/Lines", rr.LineStrips3D(hallucinated_lines, colors=[255, 0, 0, 80]))

    # Missed ground truth (Blue points and lines)
    missed_3d_pts = p3d_kpts[gt_matches0[idx_missed]]
    missed_lines = [[cam_center, pt] for pt in missed_3d_pts]
    rr.log("world/Ground_Truth/Missed/Points", rr.Points3D(missed_3d_pts, colors=[0, 150, 255], radii=0.06))
    rr.log("world/Ground_Truth/Missed/Lines", rr.LineStrips3D(missed_lines, colors=[0, 150, 255, 80]))

    # Save .rrd file
    output_filename = f"viz_{scene}_{method_name}_{query_stem}.rrd"
    rr.save(output_filename)
    logger.info(f"Rerun .rrd file visualization saved to {output_filename}")

def visual_flat_sfm(res, q_kpts, p3d_flat_kpts, img_query, p3d_colors, flat_w, flat_h, scene, method):
    logger.info(f"Generating 2D Flat SfM visualization for {method}...")
    
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

    # Add method to the filename!
    match_filename = f"flat_matches_scene_{scene}_{method}.png"
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
            
        # Add method to the filename!
        prune_filename = f"flat_pruning_scene_{scene}_{method}.png"
        plt.savefig(prune_filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
    logger.info(f"Saved 2D Flat SfM images: {match_filename} & {prune_filename}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Matches in Rerun")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth maps")
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['NN', 'RR', 'RN', 'PR', 'TRAIN', 'ADAPT'], 
                        help="Matching method to evaluate: NN, RR, RN, PR, TRAIN or ADAPT")
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="Path to trained network weights (Only required if method is TRAIN or ADAPT)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = args.scene
    method = args.method
    logger.info(f"Starting Evaluation & Visualization for Scene {scene} using method: {method}")

    # Select a random query
    query_names_file = args.query_dir / scene / "query_image_names_clean.txt"
    with open(query_names_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    query_name = random.choice(queries)
    ref_name = get_most_similar_ref(query_name, args.covisibility_dir / scene / "most_similar_pairs.txt")
    logger.info(f"Query: {query_name} | Reference: {ref_name}")

    # Load query image
    img_query_pil = Image.open(args.dataset / scene / "images" / query_name)
    q_img_size = [img_query_pil.width, img_query_pil.height]

    # Load features
    with h5py.File(args.sfm_dir / scene / "feats-superpoint-n2048.h5", "r") as f:
        q_kpts = f[query_name]["keypoints"][:]
        q_desc = f[query_name]["descriptors"][:]

    # Load visible 3d points
    sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)

    with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
        visible_p3d = pickle.load(f)[query_name]["unique_points"]

    p3d_desc, p3d_kpts, raw_colors = [], [], []

    with h5py.File(args.covisibility_dir / scene / "points3D_feats_cache.h5", "r") as f:
        for pid in visible_p3d:
            pid_int = int(pid)
            pid_str = str(pid)
            if pid_str in f and pid_int in reconstruction.points3D:
                p3d_desc.append(f[pid_str]["descriptors"][:].reshape(256))
                p3d_kpts.append(f[pid_str]["keypoints"][:].reshape(3))
                raw_colors.append(reconstruction.points3D[pid_int].color)
    if not p3d_kpts:
        logger.error("No valid 3D coordinates/features found.")
        return

    p3d_desc = np.vstack(p3d_desc).T 
    p3d_kpts = np.vstack(p3d_kpts)   
    raw_pts_np = p3d_kpts.copy() 
    # Normalize colors for the flat image projection (0 to 1)
    raw_colors_np = np.vstack(raw_colors) / 255.0

    # Calculate ground truth
    query_cams = load_query_cams(args.query_dir / scene / "query_image_cameras.txt")
    camera = query_cams[query_name]
    with h5py.File(args.depth_dir / scene / f"{Path(query_name).stem}.h5", 'r') as f:
        depth_map = f['depth'][:]
        
    gt_matches0, _ = compute_ground_truth_matches(
        {"keypoints": q_kpts}, {"keypoints": p3d_kpts}, camera, depth_map
    )

    # Load Reference Camera Pose from SfM
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
    ref_R = qvec2rotmat(ref_image_obj.qvec)
    ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
    ref_cam_obj = cameras[ref_image_obj.camera_id]

    # Initialize standard LightGlue for the projection baselines
    if method in ["RR", "RN", "PR"]:
        logger.info("Initializing standard LightGlue for baseline evaluation...")
        baseline_matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    if method == "TRAIN":
        logger.info("Loaded trained LightGlu3D model for evaluation...")
        lightglu3d_matcher = load_trained_lightglu3d(args.checkpoint, device)
    if method == "ADAPT":
        logger.info("Loaded trained LightGlue_Adapt model for evaluation...")
        lightglu3_adapt_matcher = load_trained_adapt(args.checkpoint, device)

    # Get the predected mathces
    if method == "NN":
        pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
    elif method == "RR":
        pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_rr_baseline(
            baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device
        )
    elif method == "RN":
        pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_rn_baseline(
            baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device
        )
    elif method == "PR":
        pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_pr_baseline(
            baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, ref_cam_obj, device
        )
    elif method == "TRAIN":
        if args.checkpoint is None:
            raise ValueError("--checkpoint must be provided when using the TRAIN method.")
        pred_matches0 = compute_trained_lightglu3d(lightglu3d_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
    elif method == "ADAPT":
        if args.checkpoint is None:
            raise ValueError("--checkpoint must be provided when using the ADAPT method.")
        pred_matches0 = compute_trained_lightglu3d(lightglu3_adapt_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
    
    # Evaluate metrics
    valid_pred = pred_matches0 > -1
    valid_gt = gt_matches0 > -1
    correct_matches = (pred_matches0 == gt_matches0) & valid_gt

    num_pred = valid_pred.sum()
    num_gt = valid_gt.sum()
    num_correct = correct_matches.sum()

    precision = num_correct / num_pred if num_pred > 0 else 0
    recall = num_correct / num_gt if num_gt > 0 else 0

    logger.info("="*30)
    logger.info(f"{method} Results:")
    logger.info(f"GT Matches:        {num_gt}")
    logger.info(f"Predicted Matches: {num_pred}")
    logger.info(f"Correct Matches:   {num_correct}")
    logger.info(f"Precision:         {precision:.4f}")
    logger.info(f"Recall:            {recall:.4f}")
    logger.info("="*30)

    # Launch 2D flat images
    if method in ["RR", "RN", "PR"]:
        img_query = np.array(img_query_pil.convert("RGB")) / 255.0
        visual_flat_sfm(res, q_kpts, p3d_flat_kpts, img_query, raw_colors_np, flat_w, flat_h, scene, method)

    # Launch rerun
    launch_rerun_visualization(
        pred_matches0=pred_matches0, 
        gt_matches0=gt_matches0,  
        q_kpts=q_kpts, p3d_kpts=p3d_kpts, 
        raw_pts_np=raw_pts_np, raw_colors_np=raw_colors_np,
        scene=scene, args=args, query_name=query_name, ref_name=ref_name, 
        camera=camera,
        ref_pose_matrix=ref_pose_matrix,
        method_name=method
    )

if __name__ == "__main__":
    main()