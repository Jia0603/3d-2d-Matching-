import pycolmap
from pathlib import Path

# 加载模型
scene = "0000"
ref_path = Path(f"/proj/vlarsson/datasets/megadepth/Undistorted_SfM/{scene}/sparse")
model = pycolmap.Reconstruction(ref_path)
model_image_names = {img.name for _, img in model.images.items()}

# 读取你生成的 pairs 文件
sfm_pairs_path = Path(f"./outputs/sfm/{scene}_netvlad/pairs-covisibility.txt")
pair_image_names = set()
with open(sfm_pairs_path, "r") as f:
    for line in f:
        pair_image_names.update(line.strip().split())

# 检查交集
missing_in_model = pair_image_names - model_image_names
print(f"{len(pair_image_names)} images found in Image Pairs")
print(f"{len(model_image_names)} images found in Reference Model")
print(f"Number of images not exist in Model: {len(missing_in_model)}")

if missing_in_model:
    print("The missing image name:", list(missing_in_model)[:5])