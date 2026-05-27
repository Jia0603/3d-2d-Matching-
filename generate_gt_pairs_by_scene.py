import h5py
from pathlib import Path
import pickle
import numpy as np
from utils import qvec2rotmat
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
torch.set_num_threads(1)

def load_query_cams(query_pose_path):

    query_pose_dict = {}
    with open(query_pose_path, 'r') as f:
        for line in f:
            item = line.strip().split()
            name = item[0]

            qvec = list(map(float, item[1:5]))      # 4 numbers
            tvec = list(map(float, item[5:8]))      # 3 numbers

            camera_id = item[8]
            model = item[9]
            width = int(item[10])
            height = int(item[11])
            params = list(map(float, item[12:]))

            query_pose_dict[name] = {
                "qvec": qvec,
                "tvec": tvec,
                "intrinsics": {
                    "camera_id": camera_id,
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params
                }
            }
    return query_pose_dict

def sample_depth_bilinear(depth_map, u, v):
    """
    depth_map: (H, W)
    u, v: arrays (N,)
    return: depth values (N,)
    """

    H, W = depth_map.shape

    # normalize to [-1,1] for grid_sample
    u_norm = 2.0 * u / (W - 1) - 1.0
    v_norm = 2.0 * v / (H - 1) - 1.0

    grid = torch.from_numpy(
        np.stack([u_norm, v_norm], axis=-1)
    ).float().unsqueeze(0).unsqueeze(0)  # (1,1,N,2)

    depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)

    sampled = F.grid_sample(
        depth_tensor,
        grid,
        align_corners=True,
        mode='bilinear'
    )

    return sampled.squeeze().numpy()

def compute_ground_truth_matches(
        query_feats, p3d_feats, camera, depth_map=None, 
        pos_reproj_thresh=3.0, pos_depth_thresh=0.1, neg_reproj_thresh=8.0, neg_depth_thresh=0.25
        ):
    """
    Vectorized PyTorch implementation of GT matching with Soft Thresholds.
    """

    IGNORE_FEATURE = -2
    UNMATCHED_FEATURE = -1

    kpts2d = query_feats["keypoints"]      # (N2D, 2)
    pts3d = p3d_feats["keypoints"]       # (N3D, 3)

    N2D = kpts2d.shape[0]
    N3D = pts3d.shape[0]

    # Initialize with -1
    matches0 = torch.full((N2D,), UNMATCHED_FEATURE, dtype=torch.long)
    matches1 = torch.full((N3D,), UNMATCHED_FEATURE, dtype=torch.long)

    if N3D == 0:
        return matches0, matches1

    # pose
    R = qvec2rotmat(camera["qvec"])
    t = np.array(camera["tvec"]).reshape(3, 1)

    # intrinsics
    params = camera["intrinsics"]["params"]
    fx, fy, cx, cy = params[:4]# assuming PINHOLE: fx fy cx cy
    width = camera["intrinsics"]["width"]
    height = camera["intrinsics"]["height"]

    # Vectorized 3D projection
    X = pts3d.T # (3, N3D)
    X_cam = R @ X + t # (3, N3D)
        
    z = X_cam[2, :]
    valid_z = z > 0
    valid_z = torch.as_tensor(valid_z).bool() 
    z = torch.as_tensor(z).float()
    # Avoid division by zero for invalid z
    z_safe = torch.where(valid_z, z, torch.ones_like(z))
        
    u = fx * (X_cam[0, :] / z_safe) + cx
    v = fy * (X_cam[1, :] / z_safe) + cy

    valid_proj = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    # Depth check arrays
    has_valid_depth = torch.zeros(N3D, dtype=torch.bool)
    rel_error = torch.full((N3D,), float('inf'))

    if depth_map is not None:
        # Only sample depth for points that landed inside the image bounds
        u_np, v_np = u[valid_proj].numpy(), v[valid_proj].numpy()
            
        if len(u_np) > 0:
            depth_real = sample_depth_bilinear(depth_map, u_np, v_np)
            depth_real_t = torch.as_tensor(depth_real).float().view(-1)
            # depth_real_t = torch.from_numpy(depth_real).float()
                
            valid_d = depth_real_t > 0
                
            z_valid = z[valid_proj]
            rel_err_valid = torch.full_like(depth_real_t, float('inf'))
            rel_err_valid[valid_d] = torch.abs(z_valid[valid_d] - depth_real_t[valid_d]) / depth_real_t[valid_d]
                
            # Scatter back to the original N3D sized arrays
            rel_error[valid_proj] = rel_err_valid
            has_valid_depth[valid_proj] = valid_d
            # Filter the points that have depth but bigger than neg_thresh
            depth_is_totally_wrong = has_valid_depth & (rel_error > neg_depth_thresh)
            valid_proj &= (~depth_is_totally_wrong)

    # Combine u,v into (N3D, 2)
    projected = torch.stack([u, v], dim=1)
    projected = torch.as_tensor(projected).float().contiguous()
    kpts2d = torch.as_tensor(kpts2d).float().contiguous()    
    # Vectorized Distance Matrix Calculation
    dist_matrix = torch.cdist(projected.unsqueeze(0), kpts2d.unsqueeze(0)).squeeze(0) # -> (N3D, N2D)
        
    # Mask out points that projected behind the camera or off-screen
    dist_matrix[~valid_proj] = float('inf')

    # Prepare for MNN assignment
    min_dist_3d_indices_for_2d = torch.argmin(dist_matrix, dim=0)
    min_dist_2d_indices_for_3d = torch.argmin(dist_matrix, dim=1)
    has_dist_mask = dist_matrix <= neg_reproj_thresh
    valid_2d_indices = torch.where(has_dist_mask)[1].unique()
        
    for idx2d in valid_2d_indices: # loop over valid 2d kpts
        # first assign all the existing mathes < neg_reproj_thresh as IGNORED
        has_dist_3d_idx = torch.where(has_dist_mask[:, idx2d])[0]
        if matches0[idx2d] == UNMATCHED_FEATURE:
            matches0[idx2d] = IGNORE_FEATURE
        for id in has_dist_3d_idx:
            if matches1[id] == UNMATCHED_FEATURE:
                matches1[id] = IGNORE_FEATURE
        min_dist_3d_idx = min_dist_3d_indices_for_2d[idx2d]
        is_mutual = (min_dist_2d_indices_for_3d[min_dist_3d_idx] == idx2d)
        if is_mutual:
        # If mutual nearest neighbours, check if assigned as STRICT
            cur_min_dist = dist_matrix[min_dist_3d_idx, idx2d]
            r_err = rel_error[min_dist_3d_idx]
            valid_d = has_valid_depth[min_dist_3d_idx]
            if valid_d: # if has depth
                if cur_min_dist <= pos_reproj_thresh and r_err <= pos_depth_thresh:
                    if (matches0[idx2d] in [UNMATCHED_FEATURE, IGNORE_FEATURE]) and \
                        (matches1[min_dist_3d_idx] in [UNMATCHED_FEATURE, IGNORE_FEATURE]):
                        matches0[idx2d] = min_dist_3d_idx
                        matches1[min_dist_3d_idx] = idx2d
            else: # if no depth provided
                if matches1[min_dist_3d_idx] == UNMATCHED_FEATURE:
                    matches1[min_dist_3d_idx] = IGNORE_FEATURE

    return matches0.numpy(), matches1.numpy()

def load_depth(depth_path):
    with h5py.File(depth_path, 'r') as f:
        depth = f['depth'][:]
    return depth

def generate_gt_for_query_list(
        query_list, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path, 
        pos_reproj_thresh=3.0, pos_depth_thresh=0.1, neg_reproj_thresh=8.0, neg_depth_thresh=0.25
        ):
    # extract SP keypoints descriptors of all the queries in one scene
    all_query_feats = {}
    query_set = set(query_list)
    with h5py.File(feats_2d_path, "r") as f_h5:
        all_keys = set(f_h5.keys())
        for img_name in query_set:
            if img_name in all_keys:
                ds = f_h5[img_name]
                all_query_feats[img_name] = {
                    "descriptors": ds["descriptors"][:],
                    "scores": ds["scores"][:],
                    "keypoints": ds["keypoints"][:]
                }

    # load 3d descriptors for all the 3d points
    points3d_feats = {}
    with h5py.File(feats_3d_path, "r") as f_h5:
        all_keys = set(f_h5.keys())
        for id in list(all_keys):
            ds = f_h5[str(id)]
            points3d_feats[str(id)] = {
                "descriptors": ds["descriptors"][:].reshape(1, 256),
                "keypoints": ds["keypoints"][:].reshape(1, 3),
                "scores": ds["scores"][:]
            }

    gt_data = {}
    for query in query_list:

        if query not in all_query_feats: 
            print(f"WARNING: {query} not found in all-query-feature list.")
            continue

        # query descriptors
        query_feats = all_query_feats[query]

        # load pose and camera of the query
        camera = query_cams[query] # qvec, tvec...

        # extract covisibility results of the query
        visible_p3d = covisibility_dict[query]["unique_points"]

        valid_p3d = [str(p) for p in visible_p3d if str(p) in points3d_feats]
        if not valid_p3d: 
            print(f"WARNING: No valid visible 3d points found for {query}.")
            continue

        # load keypoints and descriptors for visible points3D 
        current_p3d_feats = {
            "descriptors": np.vstack([points3d_feats[p]["descriptors"] for p in valid_p3d]),
            "keypoints": np.vstack([points3d_feats[p]["keypoints"] for p in valid_p3d]),
            "scores": [points3d_feats[p]["scores"] for p in valid_p3d]
        }

        # reproject points3d to get GT
        depth_map = load_depth(depth_path / f"{Path(query).stem}.h5")
        matches0, matches1 = compute_ground_truth_matches(
            query_feats, current_p3d_feats, camera, depth_map, 
            pos_reproj_thresh, pos_depth_thresh, neg_reproj_thresh, neg_depth_thresh
        )
        # TODO: Wash data here, only keep matches > 0
        
        gt_data[query] = {
                "keypoints0": query_feats["keypoints"], # shape (N,2))
                "descriptors0": query_feats["descriptors"].T, # to shape(N,D)
                "keypoints1": current_p3d_feats["keypoints"], # shape (M,3)
                "descriptors1": current_p3d_feats["descriptors"],# shape(M,D)
                "matches0": matches0, # shape(N,), matched 3D point index or -1
                "matches1": matches1, # shape(M,), matched 2D keypoint index or -1
        }
    
    return gt_data

if __name__ == "__main__":

    pos_reproj_thresh=3.0
    neg_reproj_thresh=8.0
    pos_depth_thresh=0.1
    neg_depth_thresh=0.25
    output_dir = Path("/proj/vlarsson/outputs")
    scene_lst_path = output_dir / "splits"
    file_path = scene_lst_path / "full_scenes.txt" 
    # file_path = "/home/x_jiagu/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean.txt"
    with open(file_path,'r')as f:
        scene_names=[item.strip() for item in f.readlines()]
    # scene_names = [] 
    # with open(scene_lst_path / "train.txt", "r") as f:
    #     for name in f.readlines():
    #         scene_names.append(name.strip())
    # with open(scene_lst_path / "val.txt", "r") as f:
    #     for name in f.readlines():
    #         scene_names.append(name.strip())
    # with open(scene_lst_path / "test.txt", "r") as f:
    #     for name in f.readlines():
    #         scene_names.append(name.strip())
    all_ratios = []
    match_2d, match_3d, ignore_2d, ignore_3d, unmatch_2d, unmatch_3d = [],[],[],[],[],[]
    for scene in tqdm(scene_names[155:]):
        scene_gt_data = {}
        query_path = output_dir / "query_sets" / scene
        query_names = query_path / "query_image_names.txt"
        query_pose = query_path / "query_image_cameras.txt"

        feats_3d_path = output_dir / "midterm_results" / scene / "points3D_feats_cache.h5" # averaged descriptors for all 3D points
        feats_2d_path = output_dir / "sfm" / scene / "feats-superpoint-n2048.h5" # cached SP descriptors
        covisibility_result_path = output_dir / "midterm_results" / scene / "covisibility_results.pkl" # covisibility results for all queries
        depth_path = Path("/proj/vlarsson/datasets/megadepth/depth_undistorted") / scene # depth maps for all queries
        # load query names to a list
        with open(query_names, 'r') as f:
            query_list = [line.strip() for line in f]

        # load covisibility result, where covisibility_results[query_image] = {'unique_images': set of img_ids,
        # 'unique_points': np.array of point3D ids, 'max_distance': float}
        with open(covisibility_result_path, "rb") as f:
            covisibility_dict = pickle.load(f)

        # load query pose infos
        query_cams = load_query_cams(query_pose)
        
        number_matches = []
        print(f"Processing Scene {scene}")

        scene_gt_data = generate_gt_for_query_list(
                query_list, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path, 
                pos_reproj_thresh, pos_depth_thresh, neg_reproj_thresh, neg_depth_thresh
                )
        
        # # Save the gt_data
        # save_path = output_dir / "gt_results" / f"{scene}_gt.pkl"
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # with open(save_path, 'wb') as f:
        #     pickle.dump(scene_gt_data, f)

        # # Release the memory
        # del scene_gt_data
        # import gc
        # gc.collect() 
        # print(f"Scene {scene} saved and memory cleared.")
        
        # -----Below is for clean query list generation-----
        query_lst_clean = []
        for query in query_list:
            matches = scene_gt_data[query]["matches0"]
            num_matches0 = np.sum(matches > -1)
            ratio = num_matches0 / np.shape(matches)[0]
            if  num_matches0 > 0: #ratio >= 0.1:
                query_lst_clean.append(query)
        print(f"{len(query_list)} queries in total originally.")
        print(f"{len(query_lst_clean)} queries that have GT matches collected.")
        # print(f"{len(query_lst_clean)} queries that have over 10% GT collected.")
        with open(query_path / "query_image_names_0_100.txt", "w") as f:
            for query_clean in query_lst_clean:
                f.write(f"{query_clean}\n")

        print(f"Clean query name saved to {query_path}  / query_image_names_0_100.txt")

        # Below is to compute the distribution of matchable GT pairs
        # query_50_100 = []
        
        # for query in query_list:
        #     matches = scene_gt_data[query]["matches0"]
        #     num_matches0 = np.sum(matches >= 0)
        #     num_unmatch_2d = np.sum(matches == -1)
        #     num_ignore_2d = np.sum(matches == -2)

        #     matches_3d = scene_gt_data[query]["matches1"]
        #     num_matches1 = np.sum(matches_3d >= 0)
        #     num_unmatch_3d = np.sum(matches_3d == -1)
        #     num_ignore_3d = np.sum(matches_3d == -2)

        #     match_2d.append(num_matches0)
        #     match_3d.append(num_matches1)
        #     ignore_2d.append(num_ignore_2d)
        #     ignore_3d.append(num_ignore_3d)
        #     unmatch_2d.append(num_unmatch_2d)
        #     unmatch_3d.append(num_unmatch_3d)

            # ratio = num_matches0 / np.shape(matches)[0]
            # all_ratios.append(ratio)
            # if ratio >= 0.5:
            #     query_50_100.append(query)
        # print(f"{len(query_list)} queries in total originally.")
        # print(f"{len(query_50_100)} queries that have over 50% overlap ratio collected.")
        # with open(query_path / "query_image_names_50_100.txt", "w") as f:
        #     for q in query_50_100:
        #         f.write(f"{q}\n")
        
        
    # plt.figure(figsize=(10, 6))
    # bins = np.arange(0, 1.1, 0.1) 

    # plt.hist(all_ratios, bins=bins, edgecolor='black', alpha=0.7)

    # plt.title('Distribution of Matchable GT Pairs Ratio (All Scenes)')
    # plt.xlabel('Match Ratio (num_matches0 / number of keypoints)')
    # plt.ylabel('Number of Queries')
    # plt.xticks(bins)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # plt.savefig('match_ratio_distribution_all_scenes_reproj_thresh_5.png')
    # plt.show()

    # print(f"Total queries processed: {len(all_ratios)}")
    # print(f"Average ratio: {np.mean(all_ratios):.4f}")

    # print(f"Number of processed queries: {len(match_2d)}")
    # print(f"Average number of matched 2D keypoints: {np.mean(match_2d):2f}")
    # print(f"Average number of matched 3D keypoints: {np.mean(match_3d):2f}")
    # print(f"Average number of ignored 2D keypoints: {np.mean(ignore_2d):2f}")
    # print(f"Average number of ignored 3D keypoints: {np.mean(ignore_3d):2f}")
    # print(f"Average number of unmatchable 2D keypoints: {np.mean(unmatch_2d):2f}")
    # print(f"Average number of unmatchable 3D keypoints: {np.mean(unmatch_3d):2f}")



