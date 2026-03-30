from pathlib import Path
from covisibility_search_pipe import most_similar_pair, covisibility_search
from generate_gt_pairs_by_scene import load_query_cams, generate_gt_for_query_list
from utils import map_img_to_points3d, map_img_name_to_id, qvec2rotmat
from hloc.utils import read_write_model as rw
import numpy as np
import matplotlib.pyplot as plt
import pickle

output_dir = Path("/proj/vlarsson/outputs")
scene_lst_path = output_dir / "splits"
scene_names = [] 
with open(scene_lst_path / "full_scenes.txt", "r") as f:
    for name in f.readlines():
        scene_names.append(name.strip())

root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
final_ratios = []
for scene in scene_names[:5]:
    print(f"Start processing covisibility search for scene: {scene}...")
    images_path = root / scene / "images" # Contains all .jpg images
    sim_pair_dir = output_dir / "midterm_results" / scene
    sim_pair_dir.mkdir(parents=True, exist_ok=True)
 
    query_path = output_dir / "query_sets" / scene
    query_names_file= query_path / "query_image_names.txt"
    query_pose = query_path / "query_image_cameras.txt"

    feats_3d_path = output_dir / "midterm_results" / scene / "points3D_feats_cache.h5" # averaged descriptors for all 3D points
    feats_2d_path = output_dir / "sfm" / scene / "feats-superpoint-n2048.h5" # cached SP descriptors
    covisibility_result_path = output_dir / "midterm_results" / scene / "covisibility_results.pkl" # covisibility results for all queries
    depth_path = Path("/proj/vlarsson/datasets/megadepth/depth_undistorted") / scene # depth maps for all queries

    # for each query
    # 1. find the most similar image: num=1. function: most_similar_pair(reference_dir, query_dir, output_dir, num)
    # 2. do covisibility search: Load SfM model for current scene, then call function: covisibility_search()
    # 3. GT generation for all querys: call function:generate_gt_for_query_list() 
    # 4. collect query names for ratio<10%; store GT result for the others
    # num+=1
    # redo 1-4
    # until all the queries have GT results saved(the collected query set is empty) or num>10

    with open(query_names_file, 'r') as f:
        remaining_queries = [line.strip() for line in f]
    
    query_cams = load_query_cams(query_pose)
    final_scene_gt_data = {}
    num = 1
    max_num = 10
    target_ratio = 0.1

    # Load SfM model
    _, images, point3D = rw.read_model(
        output_dir / "sfm" / scene / "sfm_superpoint+lightglue", ext=".bin"
        )
    
    while len(remaining_queries) > 0 and num <= max_num:
        print(f"\n[Iteration num={num}] Processing {len(remaining_queries)} queries...")
        current_query_list = query_path / f"tmp_queries_num.txt"
        with open(current_query_list, 'w') as f:
            for q in remaining_queries:
                f.write(f"{q.strip()}\n")
        # return a dict: {query_name: [top_k_image_names]}
        sim_pairs = most_similar_pair(
            reference_dir=images_path, query_dir=images_path, query_list=current_query_list,
            output_dir=sim_pair_dir, num_sim_img=num
            )
        # for each remain query, do covisibility search 
        covisibility_dict = {}        
        for query_image, matched_images in sim_pairs.items():
            matched_image = matched_images[-1]
            points3d_level = map_img_to_points3d(matched_image, images)

            img = images[map_img_name_to_id(matched_image, images)]
            R, t = qvec2rotmat(img.qvec), img.tvec
            camera_center = -R.T @ t   

            unique_images, unique_points, max_distance = covisibility_search(
                points3d_level=points3d_level, images=images, points3D=point3D,
                camera_pos=camera_center, pruning=0.35, max_points=8192
            )
            covisibility_dict[query_image] = {
                'unique_images': unique_images,
                'unique_points': unique_points,
                'max_distance': max_distance
            }
        # generate GT for current remain queries
        current_gt_data = generate_gt_for_query_list(
            remaining_queries, feats_2d_path, feats_3d_path, 
            query_cams, covisibility_dict, depth_path
        )
        # check ratio: save gt for qualified queries, collect the others for the next turn
        next_remaining_queries = []
        for q_name in remaining_queries:
            matches = current_gt_data[q_name]["matches0"]
            num_matches0 = np.sum(matches != -1)
            total_feats = np.shape(matches)[0]
            ratio = num_matches0 / total_feats if total_feats > 0 else 0

            if ratio >= target_ratio:
                final_scene_gt_data[q_name] = current_gt_data[q_name]
                final_ratios.append(ratio)
            else:
                next_remaining_queries.append(q_name)
        
        print(f"Iteration {num} done. Success: {len(remaining_queries)-len(next_remaining_queries)}, "
              f"Remaining: {len(next_remaining_queries)}")
        
        remaining_queries = next_remaining_queries
        num += 1

    if len(remaining_queries) > 0:
        print(f"Warning: {len(remaining_queries)} queries failed to reach {target_ratio} ratio after {max_num} tries.")
        print(f"Five examples of the unqualified queries: {remaining_queries[:5]}")

    # # save gt results for current scene
    # save_path = output_dir / "midterm_results" / scene / "final_gt_results.pkl"
    # with open(save_path, "wb") as f:
    #     pickle.dump(final_scene_gt_data, f)

plt.figure(figsize=(10, 6))
bins = np.arange(0, 1.1, 0.1) 

plt.hist(final_ratios, bins=bins, edgecolor='black', alpha=0.7)

plt.title('Distribution of Matchable GT Pairs Ratio (After R2G2)')
plt.xlabel('Match Ratio (num_matches0 / number of keypoints)')
plt.ylabel('Number of Queries')
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('match_ratio_distribution.png')
plt.show()

print(f"Total qualified queries: {len(final_ratios)}")
print(f"Average ratio: {np.mean(final_ratios):.4f}")
    
