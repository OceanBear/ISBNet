import torch
data = torch.load('dataset/scannetv2/test/scene0807_00_inst_nostuff.pth')
print(data)  # 应输出 (coords, colors) 二元组
