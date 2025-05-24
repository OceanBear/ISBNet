import os, glob
import numpy as np
import torch

# --- 修改这两个路径 ---
BIN_FOLDER = "/mnt/g/Summer-Project/ISBNet/your_house_dataset"
OUT_FOLDER = BIN_FOLDER      # 可以直接覆盖，也可以写成一个新的目录

os.makedirs(OUT_FOLDER, exist_ok=True)

for bin_path in glob.glob(os.path.join(BIN_FOLDER, "*.bin")):
    # 假设你的 .bin 是 float32 串，4 列：x,y,z,intensity 或 rgb
    data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = torch.from_numpy(data[:, :3]).float()
    # 如果第四列是灰度或反射强度，就重复成三通道；如果它本身是 RGB，请自行拆成 3 列
    intensity = data[:, 3:4]
    rgb = torch.cat([torch.from_numpy(intensity)]*3, dim=1).float()

    N = xyz.shape[0]
    # 我们这里给语义、实例 label 都填 0（background），测试时模型只用 xyz+rgb 做推理
    semantic_label = torch.zeros(N, dtype=torch.int64)
    instance_label = torch.zeros(N, dtype=torch.int64)

    stem = os.path.splitext(os.path.basename(bin_path))[0]
    out_path = os.path.join(OUT_FOLDER, f"{stem}_inst_nostuff.pth")  # 和原代码默认 suffix 对应
    torch.save((xyz, rgb, semantic_label, instance_label), out_path)
    print(f"Saved {out_path}")
