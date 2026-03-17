from pathlib import Path

root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
scene_names = sorted([p.name for p in root.iterdir() if p.is_dir()])

num_scene = 72
train_idx, val_idx = [round(num_scene * r ) for r in [0.7, 0.9]]
print(f"{train_idx} scences in training set")
print(f"{val_idx - train_idx} scences in validation set")
print(f"{num_scene - val_idx} scences in training set")

train_path = Path(f"/proj/vlarsson/outputs/splits/train_{num_scene}.txt")
val_path = Path(f"/proj/vlarsson/outputs/splits/val_{num_scene}.txt")
test_path = Path(f"/proj/vlarsson/outputs/splits/test_{num_scene}.txt")

with open(train_path, "w") as f:
    for scene in scene_names[:train_idx]:
        f.write(scene + "\n")

with open(val_path, "w") as f:
    for scene in scene_names[train_idx:val_idx]:
        f.write(scene + "\n")

with open(test_path, "w") as f:
    for scene in scene_names[val_idx:num_scene]:
        f.write(scene + "\n")

