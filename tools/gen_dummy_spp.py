import os, glob
import numpy as np
import torch

DATA_ROOT = "/mnt/g/Summer-Project/ISBNet/your_house_dataset"
SPP_DIR = os.path.join(DATA_ROOT, "superpoints")
os.makedirs(SPP_DIR, exist_ok=True)

for bin_path in glob.glob(os.path.join(DATA_ROOT, "*_inst_nostuff.pth")):
    # 先 load 点云来拿 N
    xyz, rgb, sem, ins = torch.load(bin_path)
    N = xyz.shape[0]
    scan_id = os.path.splitext(os.path.basename(bin_path))[0].replace("_inst_nostuff","")
    dummy_spp = torch.zeros(N, dtype=torch.int64)
    torch.save(dummy_spp, os.path.join(SPP_DIR, f"{scan_id}.pth"))
    print(f"→ dumped dummy superpoints for {scan_id}, N={N}")
