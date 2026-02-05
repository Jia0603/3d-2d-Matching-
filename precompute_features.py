import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
from hloc.utils import read_write_model as rw

def extract_3d_descriptors(points3D, images, h5_path):
    """
    Compute the averaged features for 3D points
    Returns: { p3d_id: {'descriptors': ..., 'keypoints': ..., 'scores': ...} }
    """

    p3d_feature_dict = {}

    with h5py.File(h5_path, "r") as f_h5:
        for p3d_id, p3d_obj in tqdm(points3D.items(), desc="Computing 3D Features"):
            observed_descriptors = []
            observed_scores = []

            for img_id, p2d_idx in zip(p3d_obj.image_ids, p3d_obj.point2D_idxs):
                if img_id not in images:
                    continue
                
                img_name =images[img_id].name 
                
                if img_name in f_h5:
                    ds = f_h5[img_name]
                    desc = ds['descriptors'][:, p2d_idx]
                    observed_descriptors.append(desc)
                    observed_scores.append(ds['scores'][p2d_idx])
            
            if len(observed_descriptors) > 0:
                avg_desc = np.mean(observed_descriptors, axis=0)
                avg_desc /= (np.linalg.norm(avg_desc) + 1e-6)
                avg_score = np.mean(observed_scores) if observed_scores else 1.0

                p3d_feature_dict[str(p3d_id)] = {
                    'descriptors': avg_desc.reshape(1, -1),# shape (1, 1, 256)
                    'keypoints': p3d_obj.xyz.reshape(1, 3), 
                    'scores': np.array([avg_score])
                }
    print("Finished 3D features comptation.")

    return p3d_feature_dict

def save_3d_features_to_h5(feature_dict, output_path):
    with h5py.File(output_path, "w") as f:
        for p3d_id, data in feature_dict.items():
            grp = f.create_group(p3d_id)
            grp.create_dataset('descriptors', data=data['descriptors'])
            grp.create_dataset('keypoints', data=data['keypoints'])
            grp.create_dataset('scores', data=data['scores'])

if __name__ == "__main__":

    root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
    scene_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    scene = scene_names[0]
    sfm_dir = Path("/proj/vlarsson/outputs/sfm") / scene / "sfm_superpoint+lightglue"
    output_dir = Path("/proj/vlarsson/outputs/midterm_results/") / scene

    _, images, points3D = rw.read_model(sfm_dir, ext=".bin")
    h5_path = sfm_dir.parent / "feats-superpoint-n2048.h5"
    p3d_feats = extract_3d_descriptors(points3D, images, h5_path)
    print(f"Extracted features for {len(p3d_feats)} 3D points.")
    # print one example
    keys = [key for key in p3d_feats.keys()]
    print(p3d_feats[keys[0]])

    save_3d_features_to_h5(p3d_feats, output_dir / "points3D_feats_cache.h5")