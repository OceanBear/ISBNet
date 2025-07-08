import os
import numpy as np
import torch
import open3d as o3d
import segmentator

def get_superpoint(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces)
    return superpoint

if __name__ == "__main__":
    os.makedirs("superpoints", exist_ok=True)

    # Process train/val scans
    scans_trainval = os.listdir("scans")
    for scan in scans_trainval:
        out_file = os.path.join("superpoints", f"{scan}.pth")
        if os.path.exists(out_file):
            print(f"Skipping {scan}, superpoint file already exists.")
            continue

        ply_file = os.path.join("scans", scan, f"{scan}_vh_clean_2.ply")
        spp = get_superpoint(ply_file).numpy()
        torch.save(spp, out_file)

    # Process test scans
    scans_test = os.listdir("scans_test")
    for scan in scans_test:
        out_file = os.path.join("superpoints", f"{scan}.pth")
        if os.path.exists(out_file):
            print(f"Skipping {scan}, superpoint file already exists.")
            continue

        ply_file = os.path.join("scans_test", scan, f"{scan}_vh_clean_2.ply")
        spp = get_superpoint(ply_file).numpy()
        torch.save(spp, out_file)


print("finished")