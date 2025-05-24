# tools/bin2pth_ascii.py
import os
import torch
import numpy as np

src = "/mnt/g/Summer-Project/ISBNet/your_house_dataset/Subsampled-Merged-ASCII-200.bin"
dst = "/mnt/g/Summer-Project/ISBNet/dataset/stpls3d/000000_inst_nostuff.pth"

coords = []
with open(src, 'r', errors='ignore') as f:
    for line in f:
        parts = line.strip().split()      # 用空白符分割
        if len(parts) < 3:
            continue
        # 只取前三列当 XYZ
        x, y, z = map(float, parts[:3])
        coords.append((x, y, z))

xyz = np.array(coords, dtype=np.float32)
print(f"Loaded {xyz.shape[0]} points.")

rgb     = np.zeros_like(xyz)
sem_lbl = np.zeros((xyz.shape[0],), np.int64)
ins_lbl = np.zeros((xyz.shape[0],), np.int64)
spp     = np.arange(xyz.shape[0], dtype=np.int64)

os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save((xyz, rgb, sem_lbl, ins_lbl, spp), dst)
print("Saved to", dst)
