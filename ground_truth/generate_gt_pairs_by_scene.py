import h5py
from pathlib import Path
import pickle
import numpy as np
from utils import qvec2rotmat
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import torch
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
        query_feats, p3d_feats, camera, depth_map=None, reproj_thresh=3.0, depth_rel_thresh=0.1
        ):
    """
    return:
        matches0: (N2D,)
        matches1: (N3D,)
    """

    kpts2d = query_feats["keypoints"]      # (N2D, 2)
    pts3d = p3d_feats["keypoints"]       # (N3D, 3)

    N2D = kpts2d.shape[0]
    N3D = pts3d.shape[0]

    matches0 = -np.ones(N2D, dtype=int)
    matches1 = -np.ones(N3D, dtype=int)

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

    # project all 3D points
    X = pts3d.T  # (3, N3D)

    X_cam = R @ X + t  # (3, N3D)

    z = X_cam[2]
    valid = z > 0 # check depth > 0

    X_cam = X_cam[:, valid]
    z = z[valid]

    u = fx * (X_cam[0] / z) + cx
    v = fy * (X_cam[1] / z) + cy

    valid_proj = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    u = u[valid_proj]
    v = v[valid_proj]
    z = z[valid_proj]

    valid_indices = np.where(valid)[0][valid_proj]

    # check depth consistency
    if depth_map is not None:

        depth_real = sample_depth_bilinear(depth_map, u, v)
        valid_depth = depth_real > 0
        rel_error = np.full_like(depth_real, np.inf)
        rel_error[valid_depth] = (
            np.abs(z[valid_depth] - depth_real[valid_depth])
            / depth_real[valid_depth]
        ).flatten()

        depth_mask = (rel_error <= depth_rel_thresh)

        u = u[depth_mask]
        v = v[depth_mask]
        z = z[depth_mask]
        valid_indices = valid_indices[depth_mask]

    projected = np.stack([u.flatten(), v.flatten()], axis=1)

    tree = cKDTree(kpts2d)
        
    # Query the tree for the nearest 2D keypoint to each projected 3D point
    # distance_upper_bound acts as an instant cutoff mask (reproj_thresh)
    dists, min_indices = tree.query(projected, distance_upper_bound=reproj_thresh)
        
    # Iterate over the valid results and assign matches
    for idx3d, min_idx, dist in zip(valid_indices, min_indices, dists):
        # cKDTree returns len(kpts2d) if no neighbor was found within the threshold
        if min_idx < N2D: 
            if matches0[min_idx] == -1:
                matches0[min_idx] = idx3d
                matches1[idx3d] = min_idx

    return matches0, matches1

def load_depth(depth_path):
    with h5py.File(depth_path, 'r') as f:
        depth = f['depth'][:]
    return depth

def generate_gt_for_query(query_list, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path):
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
            query_feats, current_p3d_feats, camera, depth_map, reproj_thresh=3.0, depth_rel_thresh=0.1
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

    output_dir = Path("/proj/vlarsson/outputs")
    scene_lst_path = output_dir / "splits"
    scene_names = [] 
    with open(scene_lst_path / "train.txt", "r") as f:
        for name in f.readlines():
            scene_names.append(name.strip())
    with open(scene_lst_path / "val.txt", "r") as f:
        for name in f.readlines():
            scene_names.append(name.strip())
    with open(scene_lst_path / "test.txt", "r") as f:
        for name in f.readlines():
            scene_names.append(name.strip())

    for scene in scene_names:
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

        scene_gt_data = generate_gt_for_query(
                query_list, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path
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
        
        # Below is for clean query list generation
        # query_lst_clean = []
        # for query in query_list:
        #     num_matches0 = np.sum(scene_gt_data[query]["matches0"] != -1)
        #     if num_matches0 > 0:
        #         query_lst_clean.append(query)
        # print(f"{len(query_list)} queries in total originally.")
        # print(f"{len(query_lst_clean)} queries that have GT collected.")
        # with open(query_path / "query_image_names_clean.txt", "w") as f:
        #     for query_clean in query_lst_clean:
        #         f.write(f"{query_clean}\n")

        # print(f"Clean query name saved to {query_path}  / query_image_names_clean.txt")
        

