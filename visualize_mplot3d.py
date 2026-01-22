import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def read_points3D(filename):
    pts = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or len(line) < 10:
                continue
            elems = line.split()
            x, y, z = map(float, elems[1:4])
            r, g, b = map(int, elems[4:7])
            pts.append([x, y, z, r, g, b])
    return np.array(pts)

file_path = Path("./megadepth_dataset/MegaDepth_SfM_v1/MegaDepth_v1_SfM/0000/sparse/manhattan/1")
data = read_points3D(file_path / "points3D.txt")

# 使用matplotlib创建3D散点图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制点云，使用实际颜色
sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                c=data[:, 3:6]/255.0,  # 归一化到[0,1]
                s=0.1, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('MegaDepth Point Cloud')

# 保存图像
plt.savefig('pointcloud_render.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved to pointcloud_render.png")
print(f"Total points: {len(data)}")