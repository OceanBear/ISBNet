import numpy as np
arr = np.fromfile("/mnt/g/Summer-Project/ISBNet/your_house_dataset/Subsampled-200.bin", dtype=np.float32)
print("total floats:", arr.size)
for cols in range(2,9):
    if arr.size % cols == 0:
        print(f"– possible columns per point: {cols}")
