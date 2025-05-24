import os
import torch
import numpy as np
import open3d as o3d

# —— 1) 读入点云 ——
# 如果你的文件其实是 PCD（binary）或 PLY，Open3D 能自动识别：
pcd = o3d.io.read_point_cloud(
    "/mnt/g/Summer-Project/ISBNet/your_house_dataset/Subsampled-Merged-ASCII-200.bin"  # 改成你的路径
)
xyz = np.asarray(pcd.points, dtype=np.float32)
print(f"Loaded {xyz.shape[0]} points.")

# —— 2) 准备 dummy RGB / labels / superpoints ——
rgb     = np.zeros_like(xyz)                   # N×3 都填 0
sem_lbl = np.zeros((xyz.shape[0],), np.int64)  # 全 0
ins_lbl = np.zeros((xyz.shape[0],), np.int64)  # 全 0
spp     = np.arange(xyz.shape[0], dtype=np.int64)  # 每点一个超点

# —— 3) 存成 ISBNet 期待的 tuple ——
dst = "/mnt/g/Summer-Project/ISBNet/dataset/stpls3d/000000_inst_nostuff.pth"
os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save((xyz, rgb, sem_lbl, ins_lbl, spp), dst)
print("Saved to", dst)
