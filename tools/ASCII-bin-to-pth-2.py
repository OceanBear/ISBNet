import numpy as np, torch

# 1) load your raw bin
#    (adjust this to your actual bin format)
pc = np.fromfile("/mnt/g/Summer-Project/ISBNet/your_house_dataset/Subsampled-200.bin", dtype=np.float32).reshape(-1,4)
xyz = pc[:,:3]
rgb = np.zeros_like(xyz)                  # give it some dummy colors
sem_lbl = np.zeros((xyz.shape[0],), np.int64)   # all zeros
ins_lbl = np.zeros((xyz.shape[0],), np.int64)   # all zeros
spp     = np.arange(xyz.shape[0], dtype=np.int64) # each point its own “superpoint”

# 2) save exactly the 5‑tuple that CustomDataset.load() expects
torch.save((xyz, rgb, sem_lbl, ins_lbl, spp),
           "/mnt/g/Summer-Project/ISBNet/dataset/stpls3d/000000_inst_nostuff.pth")
