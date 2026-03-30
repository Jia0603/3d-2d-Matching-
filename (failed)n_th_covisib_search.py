from covisibility_search_pipe import most_similar_pair, covisibility_search
from hloc.utils import read_write_model as rw
from utils import map_img_to_points3d, map_img_name_to_id, qvec2rotmat
import pickle
from pathlib import Path

def n_th_covisib_results(images_path, output_dir, sfm_dir, n):

    # Find the most similar images for each query image in Scenexxxx
    matched_pairs_dict = most_similar_pair(
        reference_dir = images_path,
        query_dir = images_path,
        query_list = output_dir.parent.parent / "query_sets" / output_dir.name / "query_image_names.txt",
        output_dir = output_dir,
        num_sim_img = n
    )

    # Load SfM model
    _, images, point3D = rw.read_model(sfm_dir)
        
    # Conduct covisibility search for each matched pair
    covisibility_results = {}
    for query_image, matched_images in matched_pairs_dict.items():
        points3d_level = []
        if query_image not in covisibility_results:
            covisibility_results[query_image] = {}
        for i in range(n):
            matched_image = matched_images[i]
            points3d_level = map_img_to_points3d(matched_image, images)
            if len(points3d_level) == 0: 
                # If the most similar image has no 3D correspondences, move to the next one
                print(f"{query_image} with matched image {i} has no visible 3D, skip.")
                continue

            img = images[map_img_name_to_id(matched_image, images)]
            R, t = qvec2rotmat(img.qvec), img.tvec
            camera_center = -R.T @ t   

            unique_images, unique_points, max_distance = covisibility_search(
                points3d_level=points3d_level,
                images=images,
                points3D=point3D,
                camera_pos=camera_center,
                pruning=0.35,
                max_points=8192
            )
            print(f"Query Image: {query_image}, Matched Image {i}: {matched_image}")
            print(f"  Unique Images Found: {len(unique_images)}")
            print(f"  Unique 3D Points Found: {len(unique_points)}")
            print(f"  Max Camera Distance: {max_distance:.2f}\n")
            covisibility_results[query_image][i] = {
                'unique_images': unique_images,
                'unique_points': unique_points,
                'max_distance': max_distance
            }
    return covisibility_results

if __name__ == "__main__":

    # Set paths
    root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
    # scene_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    scene = "0003"
    n = 3
    output_dir = Path("/proj/vlarsson/outputs/midterm_results/") / scene
    sfm_dir = Path("/proj/vlarsson/outputs/sfm") / scene / "sfm_superpoint+lightglue"
    images_path = root / scene / "images" # Contains all .jpg images

    print(f"Start processing {n}_th covisibility search for Scene {scene}...")
    covisibility_results_n = n_th_covisib_results(images_path, output_dir, sfm_dir, n)

    # Save n_th covisibility results
    with open(output_dir / f"covisibility_results_{n}_th.pkl", "wb") as f:
        pickle.dump(covisibility_results_n, f)