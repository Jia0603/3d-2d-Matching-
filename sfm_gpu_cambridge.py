from pathlib import Path
import torch
import torch.multiprocessing as mp
from hloc import extract_features, match_features, pairs_from_retrieval, pairs_from_covisibility

root = Path("/proj/vlarsson/datasets/cambridge")
outputs = Path("/proj/vlarsson/outputs_cambridge/sfm/")

feature_conf = {
    'output': 'feats-superpoint-n2048',
    'model': {
        'name': 'superpoint',
        'nms_radius': 4,
        'max_keypoints': 2048, 
    },
    'preprocessing': {
        'grayscale': True,
        'resize_max': 1024, 
        "resize_force": False,
    },
}

# retrieval_conf = extract_features.confs["netvlad"]
matcher_conf = match_features.confs["superpoint+lightglue"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scene_names = ["StMarysChurch", "KingsCollege", "OldHospital", "ShopFacade", "GreatCourt"]

def process_scene(scene): # change the slice to process more scenes
    print(f"Start processing scene: {scene}...")

    images_path = root / scene
    image_list = []
    # read database images
    with open(root / scene / 'dataset_train.txt', 'r') as f:
        lines = f.readlines()[3:]
        for line in lines:
            data = line.split()
            img_path = data[0]
            image_list.append(img_path)

    output_dir = outputs / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using GPU for processing feature extraction and matching.")
    sfm_pairs = output_dir / "pairs-covisib20.txt"
    reference_model = root / "CambridgeLandmarks_Colmap_Retriangulated_1024px" / scene / "model_train"

    # Step 0: Get image pairs from dataset info: skip all queries
    # retrieval_path = extract_features.main(retrieval_conf, images_path, output_dir, image_list)
    # pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=50)
    pairs_from_covisibility.main(reference_model, sfm_pairs, num_matched=20)

    # Step 1: Feature extraction
    feature_path = extract_features.main(feature_conf, images_path, output_dir)

    # Step 2: Pairwise matching
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], output_dir)

    print(f"Secene {scene} feature extraction and matching on GPU completed.")

def worker(scene, semaphore):
    with semaphore:
        process_scene(scene)


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    semaphore = mp.Semaphore(4)
    processes = []

    for scene in scene_names[1:]:
        p = mp.Process(target=worker, args=(scene, semaphore))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

